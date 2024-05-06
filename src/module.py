from typing import Tuple

import torch
import torch.nn as nn
from torch import LongTensor, Tensor

from .blocks import (
    GLMultiHeadAttention,
)
from .config import ModelConfig, ETCAttentionConfig
from . import utils


class ETCLayer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        l2l_config: ETCAttentionConfig,
        l2g_config: ETCAttentionConfig,
        g2g_config: ETCAttentionConfig,
        g2l_config: ETCAttentionConfig,
        *args,
        dropout: float = 0.1,
        padding_idx: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(d_model, nhead, *args, dropout=dropout, **kwargs)
        # override `MultiHeadAttention` with `GLMultiHeadAttention`
        self.self_attn = GLMultiHeadAttention(
            d_model,
            nhead,
            l2l_config,
            l2g_config,
            g2g_config,
            g2l_config,
            # dropout=dropout, TODO?
            padding_idx=padding_idx,
        )

    def forward(
        self,
        src: Tensor,
        l2l_rel_pos_ids: LongTensor,
        l2g_rel_pos_ids: LongTensor,
        g2g_rel_pos_ids: LongTensor,
        g2l_rel_pos_ids: LongTensor,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                l2l_rel_pos_ids,
                l2g_rel_pos_ids,
                g2g_rel_pos_ids,
                g2l_rel_pos_ids,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    l2l_rel_pos_ids,
                    l2g_rel_pos_ids,
                    g2g_rel_pos_ids,
                    g2l_rel_pos_ids,
                )
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: Tensor,
        l2l_rel_pos_ids: LongTensor,
        l2g_rel_pos_ids: LongTensor,
        g2g_rel_pos_ids: LongTensor,
        g2l_rel_pos_ids: LongTensor,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            l2l_rel_pos_ids,
            l2g_rel_pos_ids,
            g2g_rel_pos_ids,
            g2l_rel_pos_ids,
        )
        return self.dropout1(x)


class ETC(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.embeds = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.padding_idx,
        )

        encoder_layer = ETCLayer(
            config.d_model,
            config.num_heads,
            config.l2l,
            config.l2g,
            config.g2g,
            config.g2l,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            padding_idx=config.padding_idx,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

    def forward(self, long_token_ids: LongTensor) -> Tuple[Tensor, Tensor]:
        """_summary_

        Args:
            long_token_ids (Tensor): B x Sl

        Returns:
            Tuple[Tensor, Tensor]:
                Tensor: B x Sl x d
                Tensor: B x Sg x d
        """
        global_token_ids, segment_ids = utils.generate_auxiliary_tokens(
            long_token_ids,
            padding_idx=self.embeds.padding_idx,
            long_to_global_ratio=self.config.long_to_global_ratio,
        )
        # global_token_ids: B x Sg
        # segment_ids: B x Sl

        (
            l2l_rel_pos_ids,
            l2g_rel_pos_ids,
            g2g_rel_pos_ids,
            g2l_rel_pos_ids,
        ) = utils.generate_relative_pos_ids(
            long_token_ids,
            global_token_ids,
            segment_ids,
            padding_idx=self.embeds.padding_idx,
            sliding_window_radius=self.config.l2l.local_attention_radius,
            sliding_rel_max_dist=self.config.g2g.rel_pos_max_distance,
            structured_rel_max_dist=self.config.g2l.rel_pos_max_distance,
            g2l_hard_masking=True,  # TODO
        )
        """
        bl := r + 1

        if directed:
            R := 2*r + 1
        else:
            R := r + 1

        Nl := ceil(Sl / bl)

        l2l_rel_pos_ids: B x Nl x bl x (3 * bl)
        l2g_rel_pos_ids: B x Sl x Sg
        g2g_rel_pos_ids: B x Sg x Sg
        g2l_rel_pos_ids: B x Sg x Sl
        """

        x_long: Tensor = self.embeds(long_token_ids)
        # x_long: B x Sl x d

        x_global: Tensor = self.embeds(global_token_ids)
        # x_global: B x Sg x d

        z = torch.cat([x_long, x_global], dim=1)
        # z: B x (Sl + Sg) x d
        for layer in self.encoder.layers:
            z = layer(
                # features
                z,
                # positional token ids
                l2l_rel_pos_ids=l2l_rel_pos_ids,
                l2g_rel_pos_ids=l2g_rel_pos_ids,
                g2g_rel_pos_ids=g2g_rel_pos_ids,
                g2l_rel_pos_ids=g2l_rel_pos_ids,
            )

        Sl = long_token_ids.shape[1]

        # z_long: B x Sl x d
        z_long = z[:, :Sl, :]

        # z_global: B x Sg x d
        z_global = z[:, Sl:, :]

        return z_long, z_global


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
            dim_feedforward=d_model * 2,
            dropout=0,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.cls_head = nn.Sequential(
        #    nn.Linear(d_model, d_model // 2, bias=False),
        #    nn.ReLU(),
        #    nn.Linear(d_model // 2, num_classes, bias=False),
        # )
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

        # logits = self.cls_head(z[:, 0, :])
        # logits: B x Nc

        return z
