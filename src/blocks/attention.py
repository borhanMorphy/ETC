from typing import Optional, Tuple

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

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads

        # r := local attention radius
        self.local_attention_radius = local_attention_radius
        # k := relative position maximum distance
        self.rel_pos_max_distance = rel_pos_max_distance
        self.num_pos_ids = 2 * rel_pos_max_distance + 1

        self.q_proj = ProjectionLayer(embed_dim, embed_dim, num_heads, identity=skip_query_projection)
        self.k_proj = ProjectionLayer(self.kdim, embed_dim, num_heads)
        self.v_proj = ProjectionLayer(self.vdim, embed_dim, num_heads)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)

        self.rel_embeds = nn.Embedding(self.num_pos_ids, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        if local_attention_radius == "segmented":
            self.generate_relative_pos_ids = SegmentedRelPosIds(
                num_heads,
                rel_pos_max_distance,
            )
        else:
            self.generate_relative_pos_ids = SlidedRelPosIds(
                num_heads,
                rel_pos_max_distance,
            )

        self.register_buffer(
            "rel_pos_ids",
            torch.arange(self.num_pos_ids).unsqueeze(0),
        )

        self.register_buffer(
            "normalizer",
            torch.tensor(self.head_dim**0.5),
        )

    @staticmethod
    def generate_local_attn_mask(
        seq_len_q: int,
        seq_len_k: int,
        local_attention_radius: int,
        device=None,
    ) -> BoolTensor:
        mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=device)
        # TODO avoid loop
        for i in range(seq_len_q):
            start = max(0, i - local_attention_radius)
            end = min(seq_len_k, i + local_attention_radius + 1)
            mask[i, start:end] = False
        return mask

    def compute_energy(
        self,
        Q: Tensor,
        K: Tensor,
        rel_pos_ids: LongTensor,
        key_padding_mask = None,
    ) -> Tensor:
        """_summary_

        Args:
            Q (Tensor): (B * h) x Sq x (dq / h)
            K (Tensor): (B * h) x Sk x (dq / h)
            rel_pos_ids: (B * h) x Sq x Sk
            key_padding_mask (BoolTensor): B x Sk

        Returns:
            Tensor: (B * h) x Sq x Sk
        """

        batch_size = Q.shape[0] // self.num_heads
        seq_len_q = Q.shape[1]
        seq_len_k = K.shape[1]

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
            ) # -> (B * h) x R x (dq / h)
        )
        # rel_embeds: (B * h) x R x (dq / h)

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

        mask = torch.zeros((batch_size, 1, seq_len_q, seq_len_k), dtype=torch.bool, device=Q.device)

        if key_padding_mask is not None:
            mask = mask | key_padding_mask.view(batch_size, 1, 1, seq_len_k)

        if self.local_attention_radius is not None:
            mask = mask | self.generate_local_attn_mask(
                seq_len_q,
                seq_len_k,
                self.local_attention_radius,
                device=Q.device,
            ).view(1, 1, seq_len_q, seq_len_k)
            # mask: B x 1 x Sq x Sk

            # avoid masking all query values which will break softmax
            mask = mask & (~mask.all(dim=3, keepdims=True))

        energy = (
            energy
            .view(
                batch_size,
                self.num_heads,
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
        key_padding_mask: Optional[BoolTensor] = None,
    ) -> Tensor:
        """

        Args:
            Q (Tensor): B x Sq x dq
            K (Tensor): B x Sk x dk
            V (Tensor): B x Sk x dv
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

        rel_pos_ids = self.generate_relative_pos_ids(
            seq_len_q,
            seq_len_k,
            batch_size,
            device=Q.device,
        )
        # rel_pos_ids: (B * h) x Sq x Sk

        energy = self.compute_energy(Q, K, rel_pos_ids, key_padding_mask=key_padding_mask)
        # energy: (B * h) x Sq x Sk

        # compute softmax over Sk dimension
        attn = F.softmax(energy, dim=2)
        # attn: (B * h) x Sq x Sk

        attn = self.dropout(attn)

        out = (
            torch.matmul(attn, V) # (B * h) x Sq x (dq / h)
            .view(
                batch_size,
                self.num_heads,
                seq_len_q,
                self.vdim // self.num_heads,
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
            dropout=dropout,
        )
        self.g2l_attn = RelativeMultiHeadAttention(
            embed_dim,
            num_heads,
            rel_pos_max_distance=1,
            #local_attention_radius="segmented",
            dropout=dropout,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[BoolTensor] = None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """_summary_

        Args:
            query (Tensor): B x (Sl + Sg) x d
            key (Tensor): B x (Sl + Sg) x d
            value (Tensor): B x (Sl + Sg) x d
            key_padding_mask (Optional[BoolTensor], optional): B x (Sl + Sg). Defaults to None.

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

        z_l = self.l2l_attn(l_query, l_key, l_value, key_padding_mask=l_key_padding_mask)
        # z_l: B x Sl x d

        z_g = self.g2l_attn(g_query, l_key, l_value, key_padding_mask=l_key_padding_mask)
        # z_g: B x Sg x d

        z = torch.cat([z_l, z_g], dim=1)
        # z: B x (Sl + Sg) x d

        return z, None