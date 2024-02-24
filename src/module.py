from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import LongTensor, Tensor, BoolTensor

from .blocks import (
    GLMultiHeadAttention,
    FixedRatioGlobalBlock,
)


class ETCLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        *args,
        sliding_window_radius: int = None,
        segment_radius: int = None,
        hard_masking: bool = False,
        global_token_ratio: int = 16,
        **kwargs,
    ):
        super().__init__(d_model, nhead, *args, **kwargs)

        # drop `self_attn` and use RelativeMultiHeadAttention

        config = {
            "g2g": {
                "rel_pos_max_distance": 3,
                "local_attention_radius": None,
            },
            "g2l": {
                "rel_pos_max_distance": 0,
                "local_attention_radius": "segmented",
                # TODO hard mask
            },
            "l2l": {
                "rel_pos_max_distance": 3,
                "local_attention_radius": 3*5,
            },
            "l2g": {
                "rel_pos_max_distance": 1,
                "local_attention_radius": "segmented",
            }
        }
        delattr(self, "self_attn")

        self.test = RelativeMultiHeadAttention(
            d_model,
            nhead,
            rel_pos_max_distance=5,
            local_attention_radius=5*5,
            dropout=0,
        )
        """
        self.local_q_proj = nn.Linear(d_model, d_model//2, bias=False)
        self.global_q_proj = nn.Linear(d_model, d_model//2, bias=False)

        self.g2g_attn = RelativeMultiHeadAttention(
            d_model // 2,
            nhead,
            config["g2g"]["rel_pos_max_distance"],
            local_attention_radius=config["g2g"]["local_attention_radius"],
            dropout=0,
            kdim=d_model,
            vdim=d_model,
            skip_query_projection=True,
        )

        self.g2l_attn = RelativeMultiHeadAttention(
            d_model // 2,
            nhead,
            config["g2l"]["rel_pos_max_distance"],
            local_attention_radius=config["g2l"]["local_attention_radius"],
            dropout=0,
            kdim=d_model,
            vdim=d_model,
            skip_query_projection=True,
        )

        self.l2l_attn = RelativeMultiHeadAttention(
            d_model // 2,
            nhead,
            config["l2l"]["rel_pos_max_distance"],
            local_attention_radius=config["l2l"]["local_attention_radius"],
            dropout=0,
            kdim=d_model,
            vdim=d_model,
            skip_query_projection=True,
        )

        self.l2g_attn = RelativeMultiHeadAttention(
            d_model // 2,
            nhead,
            config["l2g"]["rel_pos_max_distance"],
            local_attention_radius=config["l2g"]["local_attention_radius"],
            dropout=0,
            kdim=d_model,
            vdim=d_model,
            skip_query_projection=True,
        )
        """

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, _: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        z = self.test(x, x, x, key_padding_mask=key_padding_mask)

        return self.dropout1(z)

    # self-attention block
    def _sa_block_future(self, x: Tensor, _: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        # key_padding_mask: B x (Sl + Sg)

        # TODO
        x_local, x_global = x[...]
        local_key_padding_mask, global_key_padding_mask = key_padding_mask[...]

        q_local = self.local_query_proj(x_local)
        # q_local: B x Sl x d/2
        q_global = self.global_query_proj(x_global)
        # q_global: B x Sg x d/2

        z_local = torch.cat([
            self.l2l_attn(q_local, x_local, x_local, key_padding_mask=local_key_padding_mask),
            # z_l2l: B x Sl x d/2
            self.l2g_attn(q_local, x_global, x_global, key_padding_mask=global_key_padding_mask),
            # z_l2g: B x Sl x d/2
        ], dim=2)
        # z_local: B x Sl x d

        z_global = torch.cat([
            self.g2g_attn(q_global, x_global, x_global, key_padding_mask=global_key_padding_mask),
            # z_g2g: B x Sg x d/2
            self.g2l_attn(q_global, x_local, x_local, key_padding_mask=local_key_padding_mask),
            # z_g2l: B x Sg x d/2
        ], dim=2)
        # z_global: B x Sg x d

        z = torch.cat([z_local, z_global], dim=1)
        # z: B x (Sl + Sg) x d

        return self.dropout1(z)

class ETC(nn.Module):
    def __init__(
        self,
        # Transformer specific params
        d_model: int,
        num_heads: int,
        vocab_size: int,
        num_layers: int = 1,
        num_classes: int = 1,

        # ETC specific params
        long_to_global_ratio: int = 16,
        add_global_cls_token: bool = False,
        rel_pos_max_distance: int = 4,
        local_attention_radius: int = 4,
        # TODO

    ):
        super().__init__()

        self.embeds = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.auxiliary_global_layer = FixedRatioGlobalBlock(
            d_model,
            long_to_global_ratio=long_to_global_ratio,
            add_cls_token=add_global_cls_token,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward=d_model*2,
            dropout=0,
            batch_first=True,
        )

        # override `MultiHeadAttention` with `GLMultiHeadAttention`
        encoder_layer.self_attn = GLMultiHeadAttention(
            d_model,
            num_heads,
            rel_pos_max_distance,
            local_attention_radius,
            long_to_global_ratio,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes, bias=False),
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

        x_global, global_padding_mask = self.auxiliary_global_layer(
            x_long_token_ids,
            padding_mask=long_padding_mask,
        )
        # x_global: B x Sg x d
        # global_padding_mask: B x Sg

        # concat on sequence dim
        x = torch.cat([x_long, x_global], dim=1)
        # x: B x (Sl + Sg) x d

        padding_mask = torch.cat([long_padding_mask, global_padding_mask], dim=1)
        # padding_mask: B x (Sl + Sg)

        z = self.encoder(x, src_key_padding_mask=padding_mask)
        # z: B x (Sl + Sg) x d

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