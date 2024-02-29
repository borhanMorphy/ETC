from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor, BoolTensor

from .relative_position import (
    SlidedRelPosIds,
    SegmentedRelPosIds,
)
from .ffn import ProjectionLayer


class RelativeMultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rel_pos_max_distance: int,
        local_attention_radius: Optional[int] = None,
        attention_type: Literal["slided", "segmented"] = "slided",
        dropout: float = 0.0,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        skip_query_projection: bool = False,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.attention_type = attention_type

        self.q_proj = ProjectionLayer(embed_dim, embed_dim, num_heads, identity=skip_query_projection)
        self.k_proj = ProjectionLayer(self.kdim, embed_dim, num_heads)
        self.v_proj = ProjectionLayer(self.vdim, embed_dim, num_heads)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        if attention_type == "segmented":
            self.rpe = SegmentedRelPosIds(
                num_heads,
                rel_pos_max_distance,
                embed_dim,
                local_attention_radius=local_attention_radius,
            )
        elif attention_type == "slided":
            self.rpe = SlidedRelPosIds(
                num_heads,
                rel_pos_max_distance,
                embed_dim,
                local_attention_radius=local_attention_radius,
            )
        else:
            assert False

        self.register_buffer(
            "normalizer",
            torch.tensor(self.rpe.head_dim**0.5),
        )

    def compute_energy(
        self,
        Q: Tensor,
        K: Tensor,
        rel_embeds: Tensor,
        rel_pos_ids: LongTensor,
        mask: BoolTensor = None,
    ) -> Tensor:
        """_summary_

        Args:
            Q (Tensor): (B * h) x Sq x (dq / h)
            K (Tensor): (B * h) x Sk x (dq / h)
            rel_embeds (Tensor): (B * h) x R x (dq / h)
            rel_pos_ids (LongTensor): (B * h) x Sq x Sk
            mask (BoolTensor): B x 1 x Sq x Sk

        Returns:
            Tensor: (B * h) x Sq x Sk
        """

        batch_size = Q.shape[0] // self.rpe.num_heads
        seq_len_q = Q.shape[1]
        seq_len_k = K.shape[1]

        # getting all possible Q relative values
        Q_rel = Q @ rel_embeds.permute(0, 2, 1)
        # Q_rel: (B * h) x Sq x R

        energy = Q @ K.permute(0, 2, 1)
        # energy: (B * h) x Sq x Sk

        rel_shift = Q_rel.gather(
            2,
            # TODO if seq len won't change, compute this only once
            rel_pos_ids,
        )
        # rel_shift: (B * h) x Sq x Sk
        # TODO get color codes for `rel_shift`

        energy = (energy + rel_shift) / self.normalizer

        energy = (
            energy
            .view(
                batch_size,
                self.rpe.num_heads,
                seq_len_q,
                seq_len_k,
            )
            .masked_fill(
                mask,
                float("-inf"),
            )
            .flatten(
                start_dim=0,
                end_dim=1,
            )
        )
        # energy: (B * h) x Sq x Sk

        return energy

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        segment_ids: LongTensor,
        key_padding_mask: Optional[BoolTensor] = None,
    ) -> Tensor:
        """

        Args:
            Q (Tensor): B x Sq x dq
            K (Tensor): B x Sk x dk
            V (Tensor): B x Sk x dv
            segment_ids (LongTensor): B x max(Sq, Sk)
            key_padding_mask: (BoolTensor): B x Sk

        Returns:
            Tensor: B x Sq x d
        """
        # TODO think about diff d_k per Q K V
        assert K.shape[1] == V.shape[1]

        batch_size, seq_len_q = Q.shape[:2]
        seq_len_k = K.shape[1]

        Q = self.q_proj(Q) # B x Sq x d -> (B * h) x Sq x (dq / h)
        K = self.k_proj(K) # B x Sk x d -> (B * h) x Sk x (dq / h)
        V = self.v_proj(V) # B x Sk x d -> (B * h) x Sk x (dq / h)

        rel_pos_ids, rel_embeds, mask = self.rpe(
            batch_size,
            seq_len_q,
            seq_len_k,
            segment_ids=segment_ids,
            key_padding_mask=key_padding_mask,
            device=Q.device,
        )
        # rel_pos_ids: (B * h) x Sq x Sk
        # rel_embeds: (B * h) x R x (dq / h)
        # mask: B x 1 x Sq x Sk

        energy = self.compute_energy(
            Q,
            K,
            rel_embeds,
            rel_pos_ids,
            mask=mask
        )
        # energy: (B * h) x Sq x Sk

        # compute softmax over Sk dimension
        attn = F.softmax(energy, dim=2)
        # attn: (B * h) x Sq x Sk

        attn = self.dropout(attn)

        out = (
            torch.matmul(attn, V) # (B * h) x Sq x (dq / h)
            .view(
                batch_size,
                self.rpe.num_heads,
                seq_len_q,
                self.vdim // self.rpe.num_heads,
            )
            .permute(0, 2, 3, 1) # -> B x Sq x (dq / h) x h
            .flatten(
                start_dim=2,
                end_dim=3,
            ) # -> B x Sq x dq
        )

        return self.output(out) # -> B x Sq x dq


class GLMultiHeadAttention(nn.Module):
    batch_first: bool = True
    _qkv_same_embed_dim: bool = False

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rel_pos_max_distance: int,
        local_attention_radius: int,
        long_to_global_ratio: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.long_to_global_ratio = long_to_global_ratio

        #self.local_q_proj = nn.Linear(embed_dim, embed_dim//2, bias=False)
        #self.global_q_proj = nn.Linear(embed_dim, embed_dim//2, bias=False)

        self.l2l_attn = RelativeMultiHeadAttention(
            embed_dim,
            num_heads,
            rel_pos_max_distance=rel_pos_max_distance,
            local_attention_radius=local_attention_radius,
            attention_type="slided",
            dropout=dropout,
        )
        self.g2l_attn = RelativeMultiHeadAttention(
            embed_dim,
            num_heads,
            rel_pos_max_distance=1,
            local_attention_radius=2,
            attention_type="segmented",
            dropout=dropout,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[BoolTensor] = None,
        attn_mask: Optional[LongTensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """_summary_

        Args:
            query (Tensor): B x (Sl + Sg) x d
            key (Tensor): B x (Sl + Sg) x d
            value (Tensor): B x (Sl + Sg) x d
            key_padding_mask (Optional[BoolTensor], optional): B x (Sl + Sg). Defaults to None.
            attn_mask (Optional[LongTensor], optional): B x Sl. Defaults to None

            # attn_mask: 0 0 0 1 1 2 2 2 2 3 3

        Returns:
            Tuple[Tensor, Optional[Tensor]]:
                Tensor: B x (Sl + Sg) x d
        """
        seq_len = query.shape[1]

        Sg = seq_len // (1 + self.long_to_global_ratio)
        Sl = Sg * self.long_to_global_ratio

        l_query, g_query = query[:, :Sl, :], query[:, Sl:, :]
        l_key, g_key = key[:, :Sl, :], key[:, Sl:, :]
        l_value, g_value = value[:, :Sl, :], value[:, Sl:, :]
        l_key_padding_mask, g_key_padding_mask = key_padding_mask[:, :Sl], key_padding_mask[:, Sl:]

        # TODO add l2g and g2g

        z_l = self.l2l_attn(
            l_query,
            l_key,
            l_value,
            attn_mask,
            key_padding_mask=l_key_padding_mask,
        )
        # z_l: B x Sl x d

        z_g = self.g2l_attn(
            g_query,
            l_key,
            l_value,
            attn_mask,
            key_padding_mask=l_key_padding_mask,
        )
        # z_g: B x Sg x d

        z = torch.cat([z_l, z_g], dim=1)
        # z: B x (Sl + Sg) x d

        return z, None