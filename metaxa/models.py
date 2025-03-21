import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp
from flash_attn.ops.triton.layer_norm import RMSNorm

from functools import partial

from datasets import KMER_ENCODING

try:
    USE_FLASH_ATTN = torch.cuda.get_device_capability()[0] >= 8
except RuntimeError:
    USE_FLASH_ATTN = False

from typing import Optional


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
        checkpointing: bool = False,
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
            n_layers,
            d_model,
            n_heads,
            dim_ff,
            use_flash_attn=USE_FLASH_ATTN,
            checkpointing=checkpointing,
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
        lens_after_cnn = (lens - self.kernel_size) // self.stride + 1
        lens_after_cnn += 1  # due to cls token

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
        checkpointing: bool = False,
    ):
        super().__init__()

        from layers import mha_encoder_custom

        MHA.forward = mha_encoder_custom

        self.use_flash_attn = use_flash_attn

        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    nhead,
                    dim_ff,
                    use_flash_attn,
                    dropout=dropout,
                    checkpointing=checkpointing,
                )
                for _ in range(num_layers)
            ]
        )

        for layer in self.layers[1:]:
            layer.mixer.rotary_emb = self.layers[0].mixer.rotary_emb

        self.dropout = nn.Dropout(p=dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states, key_padding_mask=None):
        residual = None

        mixer_kwargs = (
            {'key_padding_mask': key_padding_mask}
            if key_padding_mask is not None
            else None
        )

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, mixer_kwargs=mixer_kwargs
            )

        hidden_states, residual = hidden_states[:, 0], residual[:, 0]
        residual = self.dropout(hidden_states) + residual
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))

        return hidden_states


def create_block(
    d_model: int,
    nhead: int,
    dim_ff: int,
    use_flash_attn: bool,
    dropout: int = 0.1,
    checkpointing: bool = False,
):
    head_dim = d_model // nhead

    mixer_cls = partial(
        MHA,
        num_heads=nhead,
        dropout=dropout,
        rotary_emb_dim=int(1.0 * head_dim),
        rotary_emb_interleaved=True,
        use_flash_attn=use_flash_attn,
        checkpointing=checkpointing,
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
