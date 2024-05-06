from typing import Optional, Tuple
import math

import torch
from torch import Tensor, LongTensor, BoolTensor
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

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

class FastRelPosIds(nn.Module):
    def __init__(
        self,
        num_heads: int,
        rel_pos_max_distance: int,
        embed_dim: int,
        local_attention_radius: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads

        if local_attention_radius is not None:
            assert local_attention_radius >= rel_pos_max_distance

        # k := relative position maximum distance
        self.rel_pos_max_distance = rel_pos_max_distance

        # r := local attention radius
        self.local_attention_radius = local_attention_radius

        # directed attention, therefore multiply by 2
        self.num_pos_ids = 2 * rel_pos_max_distance + 1

        self.head_dim = embed_dim // num_heads

        self.rel_embeds = nn.Embedding(self.num_pos_ids, embed_dim)
        self.register_buffer(
            "rel_pos_ids",
            torch.arange(self.num_pos_ids).unsqueeze(0),
        )

    def forward(
        self,
        batch_size: int,
        seq_len_q: int,
        seq_len_k: int,
        segment_ids: LongTensor = None,
        key_padding_mask: BoolTensor = None,
        device=None,
    ) -> Tuple[LongTensor, Tensor, BoolTensor]:
        """_summary_

        Args:
            batch_size (int): number of samples in the batch
            seq_len_q (int): Sequence length of query values as Sq
            seq_len_k (int): Sequence length of key values as Sk
            segment_ids (LongTensor): B x max(Sq, Sk)
            key_padding_mask: (BoolTensor): B x Sk
            device (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple[LongTensor, Tensor, BoolTensor]:
                LongTensor: (B * h) x nbq x bl x (3 * bl)
                Tensor: (B * h) x R x (dq / h)
                BoolTensor: B x nbq x bl x (3 * bl)
        """
        assert seq_len_q == seq_len_k

        seq_len = seq_len_q

        bl = self.local_attention_radius + 1
        nbq = math.ceil(seq_len / bl)

        mask = torch.zeros((batch_size, nbq, bl, 3 * bl), dtype=torch.bool, device=device)

        key_padding_left = bl
        key_padding_right = bl + (nbq * bl - seq_len)

        i = torch.arange(0, 3*bl, device=device)
        j = torch.arange(0, bl, device=device)

        rel_pos_ids = (
            (i - bl)
            .repeat(bl, 1) - j.view(bl, 1)
        )
        mask |= (
            (rel_pos_ids.abs() > self.local_attention_radius)
            .view(
                1,
                1,
                bl,
                3 * bl,
            )
            # sliding_window_mask: 1 x 1 x bl x (3 * bl)
        )

        rel_pos_ids = (
            rel_pos_ids
            .clip(
                min=-self.rel_pos_max_distance,
                max=self.rel_pos_max_distance,
            )
            + self.rel_pos_max_distance
        ).repeat(
            batch_size * self.num_heads,
            nbq,
            1,
            1,
        )
        # rel_pos_ids: (B * h) x nbq x bl x (3 * bl)

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

        if key_padding_mask is not None:
            sliding_ids = (
                torch.arange(0, 3*bl, device=device)
                .view(1, 3*bl)
                .repeat(nbq, 1)
            ) # sliding_ids: nbq x (3 * bl)

            sliding_ids += (
                torch.arange(0, nbq*bl, bl, device=device)
                .view(nbq, 1)
            )

            key_padding_mask = F.pad(
                key_padding_mask,
                (key_padding_left, key_padding_right),
                mode="constant",
                value=True,
            )

            # key_padding_mask: B x (pad_left + S + pad_right)
            key_padding_mask = (
                key_padding_mask[:, sliding_ids]
                .view(
                    batch_size,
                    nbq,
                    1,
                    3 * bl,
                )
            )
            # key_padding_mask: B x nbq x 1 x (3 * bl)
            mask |= key_padding_mask

            # avoid masking all query values, otherwise it will break softmax
            mask = mask & (~mask.all(dim=3, keepdims=True))

        return rel_pos_ids, rel_embeds, mask


class RelativePositionLayer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        rel_pos_max_distance: int,
        embed_dim: int,
        directed_relative_position: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads

        # k := relative position maximum distance
        self.rel_pos_max_distance = rel_pos_max_distance

        self.directed_relative_position = directed_relative_position

        if self.directed_relative_position:
            # directed attention, therefore multiply by 2
            self.num_pos_ids = 2 * rel_pos_max_distance + 1
        else:
            # undirected attention, mirror both directions
            self.num_pos_ids = rel_pos_max_distance + 1

        # adding the padding
        self.num_pos_ids += 1

        self.head_dim = embed_dim // num_heads

        self.rel_embeds = nn.Embedding(self.num_pos_ids, embed_dim, padding_idx=padding_idx)
        self.register_buffer(
            "rel_pos_ids",
            torch.arange(self.num_pos_ids).unsqueeze(0),
        )

    def forward(
        self,
        query_proj: Tensor,
        rel_token_ids: LongTensor,
    ) -> LongTensor:
        """TODO

        Args:
            query_proj (Tensor): (B * h) x Sq x (dq / h)
            rel_token_ids (LongTensor): B x Sq x Sk

        Returns:
            Tensor: (B * h) x Sq x Sk
        """
        batch_size = rel_token_ids.shape[0]
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

        # (B * h) x R x (dq / h)
        Q_rel = query_proj @ rel_embeds.permute(0, 2, 1)
        # Q_rel: (B * h) x Sq x R

        rel_shift = Q_rel.gather(
            2,
            (
                rel_token_ids
            )
            # B x Sq x Sk -> B x 1 x Sq x Sk
            .unsqueeze(1)
            # B x 1 x Sq x Sk -> B x h x Sq x Sk
            .repeat(1, self.num_heads, 1, 1)
            # B x h x Sq x Sk -> (B * h) x Sq x Sk
            .flatten(start_dim=0, end_dim=1)
        )
        # rel_shift: (B * h) x Sq x Sk
        # TODO get color codes for `rel_shift`

        return rel_shift
