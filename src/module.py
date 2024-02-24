from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import LongTensor, Tensor, BoolTensor

from .blocks import (
    LocalMultiHeadAttention,
    GlobalMultiHeadAttention,
    FixedRatioGlobalBlock,
)
from .mha import RelativeMultiHeadAttention
from .blocks.attention import SlidingWindowAttention


class GlobalLocalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sliding_window_radius: int,
        segment_radius: int,
        hard_masking: bool = False,
        global_token_ratio: int = 16,
        # TODO dropout
    ):
        super().__init__()

        self.global_token_ratio = global_token_ratio
        self.local_layer = LocalMultiHeadAttention(
            d_model, num_heads, sliding_window_radius, segment_radius
        )
        self.global_layer = GlobalMultiHeadAttention(
            d_model, num_heads, sliding_window_radius, segment_radius,
            hard_masking=hard_masking,
        )

    def forward(self, x: Tensor, padding_mask = None) -> Tensor:
        """

        Args:
            x (Tensor): B x (Sl + Sg) x d
            padding_mask (BoolTensor): B x (Sl + Sg)

        Returns:
            Tensor: B x (Sl + Sg) x d
        """
        # x_long: Tensor, x_global: Tensor, segment_ids: LongTensor

        # S = Sl + Sg
        # Sg = Sl / ratio

        # Sl := (S * ratio) / (1 + ratio)
        # Sg := S - Sl

        S = x.shape[1]

        Sl = (S * self.global_token_ratio) // (1 + self.global_token_ratio)
        Sg = S - Sl

        x_long = x[:, :Sl, :]
        # x_long (Tensor): B x Sl x d

        x_global = x[:, Sl:, :]
        # x_long (Tensor): B x Sg x d

        long_padding_mask = padding_mask[:, :Sl]
        global_padding_mask = padding_mask[:, Sl:]

        segment_ids = torch.arange(Sl) // self.global_token_ratio
        # segment_ids: Sl

        # TODO find a better way
        l2l_padding_mask = long_padding_mask

        l2g_padding_mask = (
            long_padding_mask.unsqueeze(2).float()\
                 @ global_padding_mask.unsqueeze(2).permute(0, 2, 1).float()
        ).bool()

        g2g_padding_mask = (
            global_padding_mask.unsqueeze(2).float()\
                 @ global_padding_mask.unsqueeze(2).permute(0, 2, 1).float()
        ).bool()

        g2l_padding_mask = (
            global_padding_mask.unsqueeze(2).float()\
                 @ long_padding_mask.unsqueeze(2).permute(0, 2, 1).float()
        ).bool()
        """
        import matplotlib.pyplot as plt
        plt.imshow(l2l_padding_mask[0].cpu())
        plt.show()
        plt.imshow(l2g_padding_mask[0].cpu())
        plt.show()
        plt.imshow(g2g_padding_mask[0].cpu())
        plt.show()
        plt.imshow(g2l_padding_mask[0].cpu())
        plt.show()
        exit(0)
        """

        z_long = self.local_layer(x_long, x_global, segment_ids,
            l2l_padding_mask=l2l_padding_mask,
            l2g_padding_mask=l2g_padding_mask,
        )
        z_global = self.global_layer(x_long, x_global, segment_ids,
            g2g_padding_mask=g2g_padding_mask,
            g2l_padding_mask=g2l_padding_mask,
        )

        return torch.cat([z_long, z_global], dim=1)


# Deprecated
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sliding_window_radius: int,
        segment_radius: int,
        hard_masking: bool = False,
        global_token_ratio: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        # TODO add dropout

        self.attn_block = GlobalLocalMultiHeadAttention(
            d_model,
            num_heads,
            sliding_window_radius,
            segment_radius,
            hard_masking=hard_masking,
            global_token_ratio=global_token_ratio,
        )
        self.layer_norm_first = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*2, bias=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model*2, d_model, bias=False),
        )
        self.layer_norm_last = nn.LayerNorm(d_model)
    
    def forward(self, x: Tensor, long_padding_mask: BoolTensor = None, global_padding_mask: BoolTensor = None) -> Tensor:
        """

        Args:
            x (Tensor): B x S x d
            long_padding_mask (BoolTensor): B x Sl
            global_padding_mask (BoolTensor): B x Sg

        Returns:
            Tensor: B x S x d
        """
        z = self.attn_block(x, long_padding_mask=long_padding_mask, global_padding_mask=global_padding_mask)

        x_out = self.layer_norm_first(x + z)

        x_out = self.layer_norm_last(x_out + self.ffn(x_out))

        return x_out


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
        self.self_attn = RelativeMultiHeadAttention(
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
        z = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)

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
        d_model: int,
        num_heads: int,
        vocab_size: int,
        sliding_window_radius: int,
        segment_radius: int,
        hard_masking: bool = False,
        global_token_ratio: int = 16,
        num_of_global_token_types: int = 1,
        num_layers: int = 1,
        num_classes = 1,
    ):
        super().__init__()

        self.embeds = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.auxiliary_global_layer = FixedRatioGlobalBlock(
            num_of_global_token_types,
            d_model,
            global_token_ratio=global_token_ratio,
        )
        encoder_layer = ETCLayer(
            d_model,
            num_heads,
            dim_feedforward=d_model*2,
            dropout=0,
            batch_first=True,

            # ETC specific keyword args
            sliding_window_radius=sliding_window_radius,
            segment_radius=segment_radius,
            hard_masking=hard_masking,
            global_token_ratio=global_token_ratio,
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

        Sl = x_long.shape[1]

        logits = self.cls_head(z[:, Sl, :])
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
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward=d_model*2,
            dropout=0,
            batch_first=True,
        )
        """
        encoder_layer = ETCLayer(
            d_model,
            num_heads,
            dim_feedforward=d_model*2,
            dropout=0,
            batch_first=True,
            sliding_window_radius=5,
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

        z = self.encoder(x, src_key_padding_mask=mask)
        # z: B x (Sl + Sg) x d

        logits = self.cls_head(z[:, 0, :])
        # logits: B x Nc

        return logits