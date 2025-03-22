import torch

from flash_attn.bert_padding import unpad_input, pad_input

from einops import rearrange


def mha_encoder_custom(
    self,
    x,
    x_kv=None,
    key_padding_mask=None,
    mixer_subset=None,
    **kwargs,
):
    """
    Arguments:
        x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
            cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
            is the is the sum of the sequence lengths in the batch.
        x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
        key_padding_mask: boolean mask, True means to keep, False means to mask out.
            (batch, seqlen). Only applicable when not using FlashAttention.
        mixer_subset: for cross-attention only. If not None, will take a subset of x
            before applying the query projection. Useful for e.g., ViT where we only care
            about the CLS token in the last layer.

        Adapted from flash-attention
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py#L587
        The original code is licensed under the BSD 3-Clause License.
    """
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

        if self.rotary_emb_dim > 0:
            qkv = self.rotary_emb(qkv)

        if self.use_flash_attn:
            qkv_unpad, indices, cu_seqlens, max_seq_len, _ = unpad_input(
                qkv, key_padding_mask
            )
            kwargs = {'cu_seqlens': cu_seqlens, 'max_seqlen': max_seq_len}

            if not self.checkpointing or not self.training:
                context = self.inner_attn(qkv_unpad, **kwargs)
            else:
                context = torch.utils.checkpoint.checkpoint(
                    self.inner_attn,
                    qkv_unpad,
                    None,
                    cu_seqlens,
                    max_seq_len,
                    use_reentrant=True,
                )

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

        if self.rotary_emb_dim > 0:
            q, kv = self.rotary_emb(q, kv)

        if not self.checkpointing:
            context = self.inner_cross_attn(q, kv, **kwargs)
        else:
            context = torch.utils.checkpoint.checkpoint(
                self.inner_cross_attn, q, kv, **kwargs
            )

    out = self.out_proj(rearrange(context, '... h d -> ... (h d)'))
    return out if not self.return_residual else (out, x)
