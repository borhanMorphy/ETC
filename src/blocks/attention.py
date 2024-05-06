from typing import Optional, Literal

import torch
import torch.nn as nn
from torch import Tensor, LongTensor

from .ffn import ProjectionLayer
from ..config import ETCAttentionConfig
from .. import utils


class RelativeMultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        skip_query_projection: bool = False,
        padding_idx: int = 0,
        rel_pos_max_distance: int = 0,
        attention_type: Literal["dense", "sparse"] = "dense",
        directed_relative_position: bool = True,
        local_attention_radius: Optional[int] = 0,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        kdim = kdim if kdim is not None else embed_dim
        vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if directed_relative_position:
            # directed attention, therefore multiply by 2
            self.num_pos_ids = 2 * rel_pos_max_distance + 1
        else:
            # undirected attention, mirror both directions
            self.num_pos_ids = rel_pos_max_distance + 1

        # adding the padding
        self.num_pos_ids += 1

        if attention_type == "dense":
            softmax_dim = 2
            block_len = 0
        elif attention_type == "sparse":
            softmax_dim = 3
            block_len = local_attention_radius + 1
        else:
            assert False

        self.q_proj = ProjectionLayer(
            embed_dim,
            embed_dim,
            num_heads,
            identity=skip_query_projection,
            block_len=block_len,
        )
        self.k_proj = ProjectionLayer(kdim, embed_dim, num_heads, block_len=block_len)
        self.v_proj = ProjectionLayer(vdim, embed_dim, num_heads, block_len=block_len)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rel_embeds = nn.Embedding(
            self.num_pos_ids, embed_dim, padding_idx=padding_idx
        )
        self.softmax = nn.Softmax(dim=softmax_dim)

        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer(
            "rel_pos_ids",
            torch.arange(self.num_pos_ids).unsqueeze(0),
        )

        self.register_buffer(
            "normalizer",
            torch.tensor(self.head_dim**0.5),
        )

    def compute_shift(
        self,
        query_proj: Tensor,
        rel_pos_ids: LongTensor,
    ) -> Tensor:
        """TODO

        Args:
            query_proj (Tensor): (B * h) x Sq x (dq / h)
            rel_pos_ids (LongTensor): B x Sq x Sk

        Returns:
            Tensor: relative position shift tensor as (B * h) x Sq x Sk
        """

        batch_size, seq_len_q, seq_len_k = rel_pos_ids.shape

        rel_embeds = (
            self.rel_embeds(self.rel_pos_ids)
            .repeat(batch_size, 1, 1)
            .view(
                batch_size,
                self.num_pos_ids,
                self.head_dim,
                self.num_heads,
            )
            .permute(0, 3, 1, 2)  # -> B x h x R x (dq / h)
            .flatten(
                start_dim=0,
                end_dim=1,
            )  # -> (B * h) x R x (dq / h)
        )
        # rel_embeds: (B * h) x R x (dq / h)

        # (B * h) x R x (dq / h)
        Q_rel = query_proj @ rel_embeds.permute(0, 2, 1)
        # Q_rel: (B * h) x Sq x R

        mask = (rel_pos_ids == self.rel_embeds.padding_idx).unsqueeze(1)
        # mask: B x 1 x Sq x Sk

        rel_shift = Q_rel.gather(
            2,
            (rel_pos_ids)
            # B x Sq x Sk -> B x 1 x Sq x Sk
            .unsqueeze(1)
            # B x 1 x Sq x Sk -> B x h x Sq x Sk
            .repeat(1, self.num_heads, 1, 1)
            # B x h x Sq x Sk -> (B * h) x Sq x Sk
            .flatten(start_dim=0, end_dim=1),
        )
        # rel_shift: (B * h) x Sq x Sk

        rel_shift = (
            rel_shift
            # (B * h) x Sq x Sk -> B x h x Sq x Sk
            .view(
                batch_size,
                self.num_heads,
                seq_len_q,
                seq_len_k,
            ).masked_fill(
                mask,
                float("-inf"),
            )
            # B x h x Sq x Sk -> (B * h) x Sq x Sk
            .flatten(start_dim=0, end_dim=1)
        )
        # rel_shift: (B * h) x Sq x Sk

        return rel_shift

    def compute_energy(
        self,
        Q: Tensor,
        K: Tensor,
        rel_shift: Tensor,
    ) -> Tensor:
        """TODO

        Args:
            Q (Tensor): (B * h) x Sq x (dq / h)
            K (Tensor): (B * h) x Sk x (dq / h)
            rel_shift (Tensor): (B * h) x Sq x Sk

        Returns:
            Tensor: (B * h) x Sq x Sk
        """
        energy = Q @ K.permute(0, 2, 1)
        # energy: (B * h) x Sq x Sk
        energy = (energy + rel_shift) / self.normalizer
        # energy: (B * h) x Sq x Sk

        return energy

    def apply_attention(self, attn: Tensor, V: Tensor, **kwargs) -> Tensor:
        """_summary_

        Args:
            attn (Tensor): (B * h) x Sq x Sk
            V (Tensor): (B * h) x Sk x (dq / h)

        Returns:
            Tensor: _description_
        """
        batch_size = attn.shape[0] // self.num_heads
        seq_len_q = attn.shape[1]
        out = (
            torch.matmul(attn, V)  # (B * h) x Sq x (dq / h)
            .view(
                batch_size,
                self.num_heads,
                seq_len_q,
                self.head_dim,
            )
            .permute(0, 2, 3, 1)  # -> B x Sq x (dq / h) x h
            .flatten(
                start_dim=2,
                end_dim=3,
            )  # -> B x Sq x dq
        )
        return out

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        rel_pos_ids: LongTensor,
    ) -> Tensor:
        """

        Args:
            Q (Tensor): B x Sq x dq
            K (Tensor): B x Sk x dk
            V (Tensor): B x Sk x dv
            rel_pos_ids (LongTensor): (B x Sq x Sk) or (B x NB x BL x (3 * BL))

        Returns:
            Tensor: B x Sq x d
        """
        seq_len = Q.shape[1]

        Q = self.q_proj(
            Q
        )  # B x Sq x d -> (B * h) x Sq x (dq / h) or (B * h) x NB x BL x (dq / h)
        K = self.k_proj(
            K
        )  # B x Sk x d -> (B * h) x Sk x (dq / h) or (B * h) x NB x BL x (dq / h)
        V = self.v_proj(
            V
        )  # B x Sk x d -> (B * h) x Sk x (dq / h) or (B * h) x NB x BL x (dq / h)

        # getting shifted queries with respect to relative positioning
        rel_shift = self.compute_shift(
            Q,
            rel_pos_ids,
        )
        # rel_shift:
        #   (B * h) x Sq x Sk
        #       or
        #   (B * h) x NB x BL x (3 * BL)

        # computing the energy based on keys and queries
        energy = self.compute_energy(
            Q,
            K,
            rel_shift,
        )
        # energy:
        #   (B * h) x Sq x Sk
        #       or
        #   (B * h) x NB x BL x (3 * BL)

        # compute softmax over last dimension
        attn = self.softmax(energy)
        # attn:
        #   (B * h) x Sq x Sk
        #       or
        #   (B * h) x NB x BL x (3 * BL)

        attn = self.dropout(attn)

        out = self.apply_attention(attn, V, seq_len=seq_len)
        # out: B x Sq x dq

        return self.output(out)  # -> B x Sq x dq


class FastRelativeMHA(RelativeMultiHeadAttention):

    def compute_shift(
        self,
        query_proj: Tensor,
        rel_pos_ids: LongTensor,
    ) -> Tensor:
        """TODO

        Args:
            query_proj (Tensor): (B * h) x NB x BL x (dq / h)
            rel_pos_ids (LongTensor): B x NB x BL x (3 * BL)

        Returns:
            Tensor: relative position shift tensor as (B * h) x NB x BL x (3 * BL)
        """
        batch_size, num_blocks, block_len, grouped_block_len = rel_pos_ids.shape

        rel_embeds = (
            self.rel_embeds(self.rel_pos_ids)
            .repeat(batch_size, 1, 1)
            .view(
                batch_size,
                self.num_pos_ids,
                self.head_dim,
                self.num_heads,
            )
            .permute(0, 3, 1, 2)  # -> B x h x R x (dq / h)
            .flatten(
                start_dim=0,
                end_dim=1,
            )  # -> (B * h) x R x (dq / h)
        )
        # rel_embeds: (B * h) x R x (dq / h)

        # (B * h) x R x (dq / h)
        Q_rel = query_proj.view(
            batch_size * self.num_heads,
            num_blocks * block_len,
            self.head_dim,
        ) @ rel_embeds.permute(0, 2, 1)

        Q_rel = Q_rel.view(
            batch_size * self.num_heads,
            num_blocks,
            block_len,
            self.num_pos_ids,
        )
        # Q_rel: (B * h) x NB x BL x R

        mask = (rel_pos_ids == self.rel_embeds.padding_idx).unsqueeze(1)
        # mask: B x 1 x NB x BL x (3 * BL)

        rel_shift = Q_rel.gather(
            3,
            (rel_pos_ids)
            # B x NB x BL x (3 * BL) -> B x 1 x NB x BL x (3 * BL)
            .unsqueeze(1)
            # B x 1 x NB x BL x (3 * BL) -> B x h x NB x BL x (3 * BL)
            .repeat(1, self.num_heads, 1, 1, 1)
            # B x h x NB x BL x (3 * BL) -> (B * h) x NB x BL x (3 * BL)
            .flatten(start_dim=0, end_dim=1),
        )

        # rel_shift: (B * h) x NB x BL x (3 * BL)
        rel_shift = (
            rel_shift
            # (B * h) x NB x BL x (3 * BL) -> B x h x NB x BL x (3 * BL)
            .view(
                batch_size,
                self.num_heads,
                num_blocks,
                block_len,
                grouped_block_len,
            ).masked_fill(
                mask,
                float("-inf"),
            )
            # B x h x NB x BL x (3 * BL) -> (B * h) x NB x BL x (3 * BL)
            .flatten(start_dim=0, end_dim=1)
        )
        # rel_shift: (B * h) x NB x BL x (3 * BL)

        return rel_shift

    def compute_energy(
        self,
        Q: Tensor,
        K: Tensor,
        rel_shift: Tensor,
    ) -> Tensor:
        """TODO

        Args:
            Q (Tensor): (B * h) x NB x BL x (dq / h)
            K (Tensor): (B * h) x NB x BL x (dq / h)
            rel_shift (Tensor): (B * h) x NB x BL x (3 * BL)

        Returns:
            Tensor: (B * h) x NB x BL x (3 * BL)
        """
        energy = Q @ (
            # (B * h) x NB x BL x (dq / h) -> (B * h) x NB x (3 * BL) x (dq / h)
            utils.blocks_to_grouped_blocks(K)
        ).permute(0, 1, 3, 2)
        # energy: (B * h) x NB x BL x (3 * BL)

        energy = (energy + rel_shift) / self.normalizer
        # energy: (B * h) x NB x BL x (3 * BL)

        return energy

    def apply_attention(self, attn: Tensor, V: Tensor, seq_len: int = None) -> Tensor:
        """_summary_

        Args:
            attn (Tensor): (B * h) x NB x BL x (3 * BL)
            V (Tensor): (B * h) x NB x BL x (d / h)

        Returns:
            Tensor: B x S x d
        """
        batch_size = attn.shape[0] // self.num_heads
        num_blocks, block_len = V.shape[1:3]

        out = (
            torch.matmul(
                attn,
                # (B * h) x NB x BL x (d / h) -> (B * h) x NB x (3 * BL) x (dq / h)
                utils.blocks_to_grouped_blocks(V),
            )  # (B * h) x NB x BL x (dq / h)
            .view(
                batch_size,
                self.num_heads,
                num_blocks,
                block_len,
                self.head_dim,
            )
            .permute(0, 2, 3, 4, 1)  # -> B x NB x BL x (dq / h) x h
            .flatten(
                start_dim=3,
                end_dim=4,
            )  # -> B x NB x BL x dq
        )
        return utils.blocks_to_seq(out, seq_len)  # B x S x dq


class GLMultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        l2l_config: ETCAttentionConfig,
        l2g_config: ETCAttentionConfig,
        g2g_config: ETCAttentionConfig,
        g2l_config: ETCAttentionConfig,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ):
        super().__init__()

        self.long_q_proj = nn.Linear(embed_dim, embed_dim // 2, bias=False)
        self.global_q_proj = nn.Linear(embed_dim, embed_dim // 2, bias=False)

        # Long Attention Layers
        self.l2l_attn = FastRelativeMHA(
            embed_dim // 2,
            num_heads,
            dropout=dropout,
            kdim=embed_dim,
            vdim=embed_dim,
            padding_idx=padding_idx,
            skip_query_projection=True,
            rel_pos_max_distance=l2l_config.rel_pos_max_distance,
            attention_type=l2l_config.attention_type,
            directed_relative_position=l2l_config.directed_relative_position,
            local_attention_radius=l2l_config.local_attention_radius,
        )
        self.l2g_attn = RelativeMultiHeadAttention(
            embed_dim // 2,
            num_heads,
            dropout=dropout,
            kdim=embed_dim,
            vdim=embed_dim,
            padding_idx=padding_idx,
            skip_query_projection=True,
            rel_pos_max_distance=l2g_config.rel_pos_max_distance,
            attention_type=l2g_config.attention_type,
            directed_relative_position=l2g_config.directed_relative_position,
            local_attention_radius=l2g_config.local_attention_radius,
        )
        # Global Attention Layers
        self.g2g_attn = RelativeMultiHeadAttention(
            embed_dim // 2,
            num_heads,
            dropout=dropout,
            kdim=embed_dim,
            vdim=embed_dim,
            padding_idx=padding_idx,
            skip_query_projection=True,
            rel_pos_max_distance=g2g_config.rel_pos_max_distance,
            attention_type=g2g_config.attention_type,
            directed_relative_position=g2g_config.directed_relative_position,
            local_attention_radius=g2g_config.local_attention_radius,
        )
        self.g2l_attn = RelativeMultiHeadAttention(
            embed_dim // 2,
            num_heads,
            dropout=dropout,
            kdim=embed_dim,
            vdim=embed_dim,
            padding_idx=padding_idx,
            skip_query_projection=True,
            rel_pos_max_distance=g2l_config.rel_pos_max_distance,
            attention_type=g2l_config.attention_type,
            directed_relative_position=g2l_config.directed_relative_position,
            local_attention_radius=g2l_config.local_attention_radius,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        l2l_rel_pos_ids: LongTensor,
        l2g_rel_pos_ids: LongTensor,
        g2g_rel_pos_ids: LongTensor,
        g2l_rel_pos_ids: LongTensor,
    ) -> Tensor:
        """TODO

        Args:
            query (Tensor): B x (Sl + Sg) x d
            key (Tensor): B x (Sl + Sg) x d
            value (Tensor): B x (Sl + Sg) x d
            l2l_rel_pos_ids (LongTensor): B x Nl x bl x (3 * bl)
            l2g_rel_pos_ids (LongTensor): B x Sl x Sg
            g2g_rel_pos_ids (LongTensor): B x Sg x Sg
            g2l_rel_pos_ids (LongTensor): B x Sg x Sl

        Returns:
            Tensor: B x (Sl + Sg) x d
        """

        B, Sl, Sg = l2g_rel_pos_ids.shape

        l_query, g_query = query[:, :Sl, :], query[:, Sl:, :]
        l_key, g_key = key[:, :Sl, :], key[:, Sl:, :]
        l_value, g_value = value[:, :Sl, :], value[:, Sl:, :]

        l_query = self.long_q_proj(l_query)
        # l_query: B x Sl x d/2

        g_query = self.global_q_proj(g_query)
        # g_query: B x Sg x d/2

        z_l2l = self.l2l_attn(
            l_query,
            l_key,
            l_value,
            l2l_rel_pos_ids,
        )
        # z_l2l: B x Sl x d/2

        z_l2g = self.l2g_attn(
            l_query,
            g_key,
            g_value,
            l2g_rel_pos_ids,
        )
        # z_l2g: B x Sl x d/2

        z_g2g = self.g2g_attn(
            g_query,
            g_key,
            g_value,
            g2g_rel_pos_ids,
        )
        # z_g2g: B x Sg x d/2

        z_g2l = self.g2l_attn(
            g_query,
            l_key,
            l_value,
            g2l_rel_pos_ids,
        )
        # z_g2l: B x Sg x d/2

        z = torch.cat(
            [
                torch.cat([z_l2l, z_l2g], dim=2),  # B x Sl x d
                torch.cat([z_g2g, z_g2l], dim=2),  # B x Sg x d
            ],
            dim=1,
        )
        # z: B x (Sl + Sg) x d

        return z
