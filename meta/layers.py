import torch
from flash_attn.layers.rotary import (
    apply_rotary_emb_qkv_,
    apply_rotary_emb_kv_,
    apply_rotary_emb_func,
)
from flash_attn.bert_padding import unpad_input, pad_input

from einops import rearrange

from typing import *


def mha_forward_fixed(
    self,
    x,
    x_kv=None,
    key_padding_mask=None,
    cu_seqlens=None,
    max_seqlen=None,
    mixer_subset=None,
    inference_params=None,
    **kwargs,
):
    """
    Arguments:
        x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
            cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
            is the is the sum of the sequence lengths in the batch.
        x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            of the sequences in the batch, used to index into x. Only applicable when using
            FlashAttention.
        max_seqlen: int. Maximum sequence length in the batch.
        key_padding_mask: boolean mask, True means to keep, False means to mask out.
            (batch, seqlen). Only applicable when not using FlashAttention.
        mixer_subset: for cross-attention only. If not None, will take a subset of x
            before applying the query projection. Useful for e.g., ViT where we only care
            about the CLS token in the last layer.
        inference_params: for generation. Adapted from Megatron-LM (and Apex)
        https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
    """
    if cu_seqlens is not None:
        assert max_seqlen is not None
        assert key_padding_mask is None
        assert self.use_flash_attn
        assert not self.dwconv
    if key_padding_mask is not None:
        assert cu_seqlens is None
        assert max_seqlen is None
        # assert not self.use_flash_attn
    if inference_params is not None:
        assert key_padding_mask is None
        assert cu_seqlens is None and max_seqlen is None
        assert not self.dwconv
    # TODO Restore this
    """kwargs = (
        {'cu_seqlens': cu_seqlens, 'max_seqlen': max_seqlen, **kwargs}
        if self.use_flash_attn
        else {'key_padding_mask': key_padding_mask, **kwargs}
    )"""
    seqlen_offset = (
        0
        if inference_params is None
        else (
            inference_params.lengths_per_sample
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
    )
    rotary_max_seqlen = (
        inference_params.max_seqlen if inference_params is not None else None
    )
    batch, seqlen = x.shape[:2]
    if not self.cross_attn and self.num_heads_kv == self.num_heads:
        assert x_kv is None and mixer_subset is None
        if not self.return_residual:
            qkv = self.Wqkv(x)
        else:
            qkv, x = self.Wqkv(x)
        if self.dwconv:
            qkv = rearrange(
                self.dwconv_qkv(rearrange(qkv, 'b s d -> b d s'))[..., :-2],
                'b d s -> b s d',
            ).contiguous()
        qkv = rearrange(
            qkv, '... (three h d) -> ... three h d', three=3, d=self.head_dim
        )
        if (
            inference_params is None
            or inference_params.seqlen_offset == 0
            or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
            or not self.use_flash_attn
        ):
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(
                    qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )
            if inference_params is None:
                if not self.checkpointing:
                    if self.use_flash_attn:
                        qkv_unpad, indices, cu_seqlens, max_seq_len = unpad_input(
                            qkv, key_padding_mask
                        )
                        kwargs = {'cu_seqlens': cu_seqlens, 'max_seqlen': max_seq_len}

                        context = self.inner_attn(qkv_unpad, **kwargs)

                        context = pad_input(
                            context,
                            indices,
                            qkv.size(0),
                            qkv.size(1),
                        )
                    else:
                        kwargs = {'key_padding_mask': key_padding_mask}
                        context = self.inner_attn(qkv, **kwargs)
                else:
                    context = torch.utils.checkpoint.checkpoint(
                        self.inner_attn, qkv, **kwargs
                    )
            else:
                context = self._update_kvcache_attention(
                    qkv[:, :, 0], qkv[:, :, 1:], inference_params
                )
        else:
            context = self._apply_rotary_update_kvcache_attention(
                qkv[:, :, 0], qkv[:, :, 1:], inference_params
            )
    else:
        if self.cross_attn:
            if not self.return_residual:
                q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
                kv = self.Wkv(x_kv if x_kv is not None else x)
            else:
                if x_kv is not None:
                    kv, x_kv = self.Wkv(x_kv)
                else:
                    kv, x = self.Wkv(x)
                q = self.Wq(x if mixer_subset is None else x[:, mixer_subset])
        else:
            assert self.num_heads_kv != self.num_heads
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            q = qkv[..., : self.num_heads * self.head_dim]
            kv = qkv[..., self.num_heads * self.head_dim :]
        q = rearrange(q, '... (h d) -> ... h d', d=self.head_dim)
        kv = rearrange(kv, '... (two hkv d) -> ... two hkv d', two=2, d=self.head_dim)
        if self.dwconv:
            q = rearrange(
                self.dwconv_q(rearrange(q, 'b s d -> b d s'))[..., :-2],
                'b d s -> b s d',
            ).contiguous()
            kv = rearrange(
                self.dwconv_kv(rearrange(kv, 'b s d -> b d s'))[..., :-2],
                'b d s -> b s d',
            ).contiguous()
        if (
            inference_params is None
            or inference_params.seqlen_offset == 0
            or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
            or not self.use_flash_attn
        ):
            if self.rotary_emb_dim > 0:
                q, kv = self.rotary_emb(
                    q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )
            if inference_params is None:
                if not self.checkpointing:
                    context = self.inner_cross_attn(q, kv, **kwargs)
                else:
                    context = torch.utils.checkpoint.checkpoint(
                        self.inner_cross_attn, q, kv, **kwargs
                    )
            else:
                context = self._update_kvcache_attention(q, kv, inference_params)
        else:
            context = self._apply_rotary_update_kvcache_attention(
                q, kv, inference_params
            )
    out = self.out_proj(rearrange(context, '... h d -> ... (h d)'))
    return out if not self.return_residual else (out, x)


def rotary_forward_fixed(
    self,
    qkv: torch.Tensor,
    kv: Optional[torch.Tensor] = None,
    seqlen_offset: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    qkv: (batch, seqlen, 3, nheads, headdim) if kv is none,
         else it's just q of shape (batch, seqlen, nheads, headdim)
    kv: (batch, seqlen, 2, nheads, headdim)
    seqlen_offset: (batch_size,) or int. Each sequence in x is shifted by this amount.
        Most commonly used in inference when we have KV cache.
        If it's a tensor of shape (batch_size,), then to update the cos / sin cache, one
        should pass in max_seqlen, which will update the cos / sin cache up to that length.
    Apply rotary embedding *inplace* to qkv and / or kv.
    """
    seqlen = qkv.shape[1]
    if max_seqlen is not None:
        self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
    elif isinstance(seqlen_offset, int):
        self._update_cos_sin_cache(
            seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype
        )

    if cu_seqlens is not None:
        apply_rotary_emb_func(
            qkv[:, 0],
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            inplace=True,
            seqlen_offsets=seqlen_offset,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.scale is None:
            apply_rotary_emb_func(
                qkv[:, 1],
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            apply_rotary_emb_func(
                qkv[:, 1],
                self._cos_k_cached,
                self._sin_k_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        return qkv
    else:
        if kv is None:
            if self.scale is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
        else:
            q = qkv
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            return q, kv
