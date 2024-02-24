from typing import Optional

import torch
from torch import Tensor, LongTensor, BoolTensor
import torch.nn as nn
import torch.nn.functional as F


class SlidedRelPosIds(nn.Module):
    def __init__(self, num_heads: int, rel_pos_max_distance: int):
        """_summary_

        Args:
            num_heads (int): how many heads the attention module have
            rel_pos_max_distance (int): maximum distance for relative position embedding
        """
        super().__init__()
        self.num_heads = num_heads
        self.rel_pos_max_distance = rel_pos_max_distance

    def forward(
        self,
        seq_len_q: int,
        seq_len_k: int,
        batch_size: int,
        device=None,
    ) -> LongTensor:
        """_summary_

        Args:
            seq_len_q (int): Sequence length of query values as Sq
            seq_len_k (int): Sequence length of key values as Sk
            batch_size (int): number of samples in the batch
            device (_type_, optional): _description_. Defaults to None.

        Returns:
            LongTensor: (B * h) x Sq x Sk
        """
        i = torch.arange(seq_len_k, device=device).repeat(seq_len_q, 1)
        j = torch.arange(seq_len_q, device=device).unsqueeze(1)
        ids = (i - j).clip(min=-self.rel_pos_max_distance, max=self.rel_pos_max_distance) + self.rel_pos_max_distance
        return ids.repeat(batch_size * self.num_heads, 1, 1)

class SegmentedRelPosIds(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        ...


class ProjectionLayer(nn.Module):
    def __init__(self, fin: int, fout: int, nheads: int, bias: bool = False, identity: bool = False):
        super().__init__()
        assert fout % nheads == 0

        self.nheads = nheads
        self.head_dim = fout // nheads
        self.nn = (
            nn.Identity()
            if identity
            else
            nn.Linear(fin, fout, bias=bias)
        )

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): B x S x d

        Returns:
            Tensor: (B * h) x S x (d / h)
        """
        batch_size, seq_len = x.shape[:2]
        return (
            self.nn(x)
            .view(
                batch_size,
                seq_len,
                self.head_dim,
                self.nheads,
            )
            .permute(0, 3, 1, 2)  # -> B x h x S x (d / h)
            .flatten(
                start_dim=0,
                end_dim=1,
            ) # -> (B * h) x S x (d / h)
        )

class RelativeMultiHeadAttention(nn.Module):
    batch_first: bool = True

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
        # TODO
        self._qkv_same_embed_dim = False#self.kdim == embed_dim and self.vdim == embed_dim

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
        if local_attention_radius != "segmented":
            self.generate_relative_pos_ids = SlidedRelPosIds(
                num_heads,
                rel_pos_max_distance,
            )
        else:
            self.generate_relative_pos_ids = SegmentedRelPosIds(
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
