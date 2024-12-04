import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp
from flash_attn.ops.triton.layer_norm import RMSNorm
from flash_attn.bert_padding import (
    index_first_axis_residual,
    pad_input,
    unpad_input,
)

from functools import partial

from datasets import KMER_ENCODING

try:
    USE_FLASH_ATTN = torch.cuda.get_device_capability()[0] >= 8

    from layers import mha_forward_fixed

    MHA.forward = mha_forward_fixed
    # Cannot use Rotary with flash attn
    # USE_FLASH_ATTN = False
except RuntimeError:
    USE_FLASH_ATTN = False

from typing import Optional


class FFBlock(nn.Module):
    def __init__(self, f_in: int, f_out: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(f_in, f_out),
            nn.GELU(),
            nn.BatchNorm1d(f_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class KMerEncodingTransformer(nn.Module):
    def __init__(self, f_in: int, f_out: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            len(KMER_ENCODING), 512, padding_idx=KMER_ENCODING['PAD']
        )

        self.encoder = TransformerEncoder(
            8, 512, 8, 2048, use_flash_attn=USE_FLASH_ATTN
        )
        self.fc = nn.Linear(512, f_out)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        x = self.embedding(x)
        x = self.encoder(x, key_padding_mask=attn_mask)

        return self.fc(x)


class CNNEncodingTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_ff: int,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        f_in = 4  # One-hot encoded sequence
        self.kernel_size = 31
        self.stride = 5

        self.embedding = nn.Conv1d(
            f_in, d_model, kernel_size=self.kernel_size, stride=self.stride, bias=False
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.encoder = TransformerEncoder(
            n_layers, d_model, n_heads, dim_ff, use_flash_attn=USE_FLASH_ATTN
        )

        self.reset_parameters()

    def forward(
        self, x: torch.Tensor, lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x.transpose(1, 2)  # N x L x 4 -> N x 4 x L
        x = self.embedding(x)  # N x 4 x L -> N x 512 x L/5
        x = x.transpose(1, 2)  # N x 512 x L/5 -> N x L/5 x 512

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        attn_mask = (
            None if lens is None else self.create_attn_mask(lens, self.cls_token.device)
        )

        x = self.encoder(x, key_padding_mask=attn_mask)

        return x

    def reset_parameters(self):
        nn.init.normal_(self.cls_token, std=1e-6)

    def create_attn_mask(self, lens: torch.Tensor, device=torch.device) -> torch.Tensor:
        lens_after_cnn = (lens - self.kernel_size) // 5 + 1
        arange = torch.arange(lens_after_cnn.max(), device=device).expand(
            (len(lens_after_cnn), -1)
        )
        attn_mask = arange < lens_after_cnn.unsqueeze(-1)

        return attn_mask


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_ff: int,
        use_flash_attn: bool,
        dropout: int = 0.1,
    ):
        super().__init__()

        self.use_flash_attn = use_flash_attn

        self.layers = nn.ModuleList(
            [
                create_block(d_model, nhead, dim_ff, use_flash_attn, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        for layer in self.layers[1:]:
            layer.mixer.rotary_emb = self.layers[0].mixer.rotary_emb

        self.dropout = nn.Dropout(p=dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states, key_padding_mask=None, subset_mask=None):
        """If subset_mask is not None, we only want output for the subset of the sequence.
        This means that we only compute the last layer output for these tokens.
        subset_mask: (batch, seqlen), dtype=torch.bool
        """
        residual = None
        # TODO Remove True
        if key_padding_mask is None or not self.use_flash_attn or True:
            mixer_kwargs = (
                {'key_padding_mask': key_padding_mask}
                if key_padding_mask is not None
                else None
            )
            for layer in self.layers:
                hidden_states, residual = layer(
                    hidden_states, residual, mixer_kwargs=mixer_kwargs
                )
            if subset_mask is not None:
                hidden_states = hidden_states[subset_mask]
        else:
            batch, seqlen = hidden_states.shape[:2]
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                hidden_states, key_padding_mask
            )
            mixer_kwargs = {'cu_seqlens': cu_seqlens, 'max_seqlen': max_seqlen_in_batch}
            if subset_mask is None:
                for layer in self.layers:
                    hidden_states, residual = layer(
                        hidden_states, residual, mixer_kwargs=mixer_kwargs
                    )
                hidden_states = pad_input(hidden_states, indices, batch, seqlen)
            else:
                for layer in self.layers[:-1]:
                    hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                if key_padding_mask is not None:
                    subset_idx = torch.nonzero(
                        subset_mask[key_padding_mask], as_tuple=False
                    ).flatten()
                    subset_seqlens = (subset_mask & key_padding_mask).sum(
                        dim=-1, dtype=torch.int32
                    )
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32),
                        (1, 0),
                    )
                else:
                    subset_idx = torch.nonzero(subset_mask, as_tuple=False).flatten()
                    subset_seqlens = subset_mask.sum(dim=-1, dtype=torch.int32)
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32),
                        (1, 0),
                    )
                hidden_states_subset, hidden_states = index_first_axis_residual(
                    hidden_states, subset_idx
                )
                # It's ok to set max_seqlen_q to be much larger
                mixer_kwargs = {
                    'x_kv': hidden_states,
                    'cu_seqlens': subset_cu_seqlens,
                    'max_seqlen': max_seqlen_in_batch,
                    'cu_seqlens_k': cu_seqlens,
                    'max_seqlen_k': max_seqlen_in_batch,
                }
                hidden_states = self.layers[-1](
                    hidden_states_subset, mixer_kwargs=mixer_kwargs
                )

        # TODO: Use subset mask from flash attention for last layer
        # TODO: IT SHOULD BE ADD + DROPOUT + LN -> Return this for KMER
        # x = self.norm(hidden_states[:, 0])

        hidden_states, residual = hidden_states[:, 0], residual[:, 0]
        residual = self.dropout(hidden_states) + residual
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

        return hidden_states


def create_block(
    d_model: int, nhead: int, dim_ff: int, use_flash_attn: bool, dropout: int = 0.1
):
    head_dim = d_model // nhead

    mixer_cls = partial(
        MHA,
        num_heads=nhead,
        dropout=dropout,
        rotary_emb_dim=int(1.0 * head_dim),
        rotary_emb_interleaved=True,
        use_flash_attn=use_flash_attn,
    )

    mlp_cls = partial(Mlp, hidden_features=dim_ff, activation=F.silu)

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=RMSNorm,
        prenorm=True,
        resid_dropout1=dropout,
        resid_dropout2=dropout,
        residual_in_fp32=True,
    )

    return block
