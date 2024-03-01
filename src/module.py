from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import LongTensor, Tensor, BoolTensor

from .blocks import (
    GLMultiHeadAttention,
    FixedRatioGlobalBlock,
)
from .config import ModelConfig

class ETC(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embeds = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.padding_idx)
        self.auxiliary_global_layer = FixedRatioGlobalBlock(
            config.d_model,
            long_to_global_ratio=config.long_to_global_ratio,
            add_cls_token=config.add_global_cls_token,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )

        # override `MultiHeadAttention` with `GLMultiHeadAttention`
        encoder_layer.self_attn = GLMultiHeadAttention(
            config.d_model,
            config.num_heads,
            config.long_to_global_ratio,
            config.l2l,
            config.l2g,
            config.g2g,
            config.g2l,
            dropout=config.dropout,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.cls_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2, bias=False),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.num_classes, bias=False),
        )


    def forward(self, x_long_token_ids: LongTensor) -> Tensor:
        """_summary_

        Args:
            x_long_token_ids (Tensor): B x Sl

        Returns:
            Tensor: B x Nc
        """
        batch_size, Sl = x_long_token_ids.shape[:2]
        Sg = Sl // self.auxiliary_global_layer.long_to_global_ratio

        long_padding_mask = x_long_token_ids == self.embeds.padding_idx
        # long_padding_mask: B x Sl

        x_long: Tensor = self.embeds(x_long_token_ids)
        # x_long: B x Sl x d

        x_global, segment_ids, global_padding_mask = self.auxiliary_global_layer(
            x_long_token_ids,
            padding_mask=long_padding_mask,
        )
        # x_global: B x Sg x d
        # segment_ids: B x Sl
        # global_padding_mask: B x Sg

        # concat on sequence dim
        x = torch.cat([x_long, x_global], dim=1)
        # x: B x (Sl + Sg) x d

        padding_mask = torch.cat([long_padding_mask, global_padding_mask], dim=1)
        # padding_mask: B x (Sl + Sg)

        z = self.encoder(x, mask=segment_ids, src_key_padding_mask=padding_mask)
        # z: B x (Sl + Sg) x d

        # TODO
        z_cls = (
            z[:, Sl:, :]
            .masked_fill(
                global_padding_mask.view(batch_size, Sg, 1),
                0,
            )
            .sum(dim=1)
        ) / (~global_padding_mask).sum(dim=1, keepdims=True)

        logits = self.cls_head(z_cls)
        # logits: B x Nc

        return logits

class VanillaTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        vocab_size: int,
        max_seq_len: int,
        num_layers: int = 1,
        num_classes: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.embeds = nn.Embedding(vocab_size, d_model, padding_idx=0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward=d_model*2,
            dropout=0,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes, bias=False),
        )
        self.pos_embeds = nn.Embedding(max_seq_len, d_model)


    def forward(self, x_long_token_ids: LongTensor) -> Tensor:
        """_summary_

        Args:
            x_long_token_ids (Tensor): B x Sl

        Returns:
            Tensor: B x Nc
        """
        mask = x_long_token_ids == 0

        x: Tensor = self.embeds(x_long_token_ids)
        # x: B x Sl x d

        pos_ids = torch.arange(x.shape[1], device=x.device)

        z = self.encoder(x + self.pos_embeds(pos_ids), src_key_padding_mask=mask)
        # z: B x (Sl + Sg) x d

        logits = self.cls_head(z[:, 0, :])
        # logits: B x Nc

        return logits