from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor, BoolTensor


class SlidedRelativePE(nn.Module):
    # Validated

    def __init__(
        self,
        hidden_dim: int,
        relative_pos_max_distance: int,
    ):
        super().__init__()

        self.relative_pos_max_distance = relative_pos_max_distance
        self.rel_embeds = nn.Embedding(
            2 * relative_pos_max_distance + 1,
            hidden_dim,
        )

        self.register_buffer(
            "rel_pos_ids",
            torch.arange(2 * relative_pos_max_distance + 1).unsqueeze(0),
        )
        self.register_buffer(
            "normalizer",
            torch.tensor(hidden_dim**0.5),
        )

    @staticmethod
    def generate_relative_pos_indexes(
        seq_len_q: int,
        seq_len_k: int,
        radius: int,
    ) -> LongTensor:
        i = torch.arange(seq_len_k).repeat(seq_len_q, 1)
        j = torch.arange(seq_len_q).unsqueeze(1)
        return (i - j).clamp(min=-radius, max=radius) + radius

    @torch.no_grad()
    def get_relative_attention(self, seq_len_q: int, seq_len_k: int) -> Tensor:
        # TODO mask
        batch_size = 1
        rel_embeds = self.rel_embeds(self.rel_pos_ids)
        # rel_embeds: 1 x (2*r+1) x d
        d = rel_embeds.shape[2]

        Q = torch.ones((batch_size, seq_len_q, d))

        # getting all possible Q relative values
        Q_rel = Q @ rel_embeds.permute(0, 2, 1)
        # Q_rel: 1 x S1 x (2*r+1)

        rel_shift = Q_rel.gather(
            2,
            # TODO if seq len won't change, compute this only once
            self.generate_relative_pos_indexes(
                seq_len_q,
                seq_len_k,
                self.relative_pos_max_distance,
            ).repeat(
                batch_size,
                1,
                1,
            ),
        )
        return rel_shift.squeeze(0)  # -> S1 x S2

    def naive_forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len_q = Q.shape[:2]
        seq_len_k = K.shape[1]

        energy = torch.zeros(batch_size, seq_len_q, seq_len_k)
        for b in range(batch_size):
            for i in range(seq_len_q):
                for j in range(seq_len_k):
                    k = max(j - i, -self.relative_pos_max_distance)
                    k = min(k, self.relative_pos_max_distance)
                    k = torch.LongTensor([k + self.relative_pos_max_distance])

                    energy[b, i, j] = (
                        Q[b, i] @ (K[b, j] + self.rel_embeds(k)).T
                    ) / self.normalizer

        # compute softmax over S2 dimension
        attn = F.softmax(energy, dim=2)
        # attn: B x S1 x S2

        return attn, attn @ V

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """

        Args:
            Q (Tensor): B x S1 x d
            K (Tensor): B x S2 x d
            V (Tensor): B x S2 x d

        Returns:
            Tuple[Tensor, Tensor]:
                attention : B x S1 x S2
                Z         : B x S1 x d
        """

        batch_size, seq_len_q = Q.shape[:2]
        seq_len_k = K.shape[1]

        rel_embeds = self.rel_embeds(self.rel_pos_ids)
        # rel_embeds: B x R x d

        # getting all possible Q relative values
        Q_rel = Q @ rel_embeds.permute(0, 2, 1)
        # Q_rel: B x S1 x R

        energy = Q @ K.permute(0, 2, 1)
        # energy: B x S1 x S2

        rel_shift = Q_rel.gather(
            2,
            # TODO if seq len won't change, compute this only once
            self.generate_relative_pos_indexes(
                seq_len_q,
                seq_len_k,
                self.relative_pos_max_distance,
            ).repeat(
                batch_size,
                1,
                1,
            ),
        )
        # rel_shift: B x S1 x S2

        energy = (energy + rel_shift) / self.normalizer

        # compute softmax over S2 dimension
        attn = F.softmax(energy, dim=2)
        # attn: B x S1 x S2

        return attn, attn @ V


class RelativePE(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        radius: int,
        segmented: bool = False,
        change_direction: bool = False,
        hard_masking: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not segmented:
            assert change_direction is False

        if hard_masking:
            assert segmented is True

        self.segmented = segmented
        self.change_direction = change_direction
        self.hard_masking = hard_masking
        # r
        self.radius = radius
        # R
        self.num_pos_ids = (radius + 1) if segmented else (2 * radius + 1)

        self.rel_embeds = nn.Embedding(self.num_pos_ids, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer(
            "rel_pos_ids",
            torch.arange(self.num_pos_ids).unsqueeze(0),
        )

        self.register_buffer(
            "normalizer",
            torch.tensor(hidden_dim**0.5),
        )

    @staticmethod
    def generate_slided_relative_pos_ids(
        seq_len: int,
        radius: int,
    ) -> LongTensor:
        i = torch.arange(seq_len).repeat(seq_len, 1)
        j = torch.arange(seq_len).unsqueeze(1)
        return (i - j).clamp(min=-radius, max=radius) + radius

    @staticmethod
    def generate_segmented_relative_pos_ids(
        segment_ids: LongTensor,
        radius: int,
        change_direction: bool = False,
    ) -> LongTensor:
        # segment_ids: (S,)

        long_seq_len = segment_ids.shape[0]
        global_seq_len = segment_ids.max() + 1

        i = segment_ids.repeat(global_seq_len, 1)

        j = torch.arange(global_seq_len).unsqueeze(1)

        rel_pos_ids = (i - j).clamp(min=-radius, max=radius).abs()

        if change_direction:
            rel_pos_ids = rel_pos_ids.T

        return rel_pos_ids

    @torch.no_grad()
    def get_relative_attention(
        self,
        seq_len: int,
        segment_ids: LongTensor,
    ) -> Tensor:
        # TODO mask
        batch_size = 1

        rel_embeds = self.rel_embeds(self.rel_pos_ids)
        # rel_embeds: 1 x R x d
        d = rel_embeds.shape[2]

        Q = torch.ones((batch_size, seq_len, d))

        # getting all possible Q relative values
        Q_rel = Q @ rel_embeds.permute(0, 2, 1)
        # Q_rel: 1 x S1 x R

        if self.segmented:
            rel_pos_ids = self.generate_segmented_relative_pos_ids(
                segment_ids,
                self.radius,
                change_direction=self.change_direction,
            )
            # rel_pos_ids: S1 x S2
        else:
            rel_pos_ids = self.generate_slided_relative_pos_ids(
                seq_len,
                self.radius,
            )
            # rel_pos_ids: S1 x S2

        rel_shift = Q_rel.gather(
            2,
            # TODO if seq len won't change, compute this only once
            rel_pos_ids.repeat(batch_size, 1, 1),
        )
        return rel_shift.squeeze(0)  # -> S1 x S2

    def compute_energy(
        self,
        Q: Tensor,
        K: Tensor,
        segment_ids: LongTensor,
        padding_mask = None,
    ) -> Tensor:
        """_summary_

        Args:
            Q (Tensor): B x S1 x d
            K (Tensor): B x S2 x d
            segment_ids (LongTensor): max(S1, S2)
            padding_mask (BoolTensor): B x S1 x S2

        Returns:
            Tensor: B x S1 x S2
        """
        batch_size, seq_len_q = Q.shape[:2]
        seq_len_k = K.shape[1]

        rel_embeds = self.rel_embeds(self.rel_pos_ids)
        # rel_embeds: B x R x d

        # getting all possible Q relative values
        Q_rel = Q @ rel_embeds.permute(0, 2, 1)
        # Q_rel: B x S1 x R

        energy = Q @ K.permute(0, 2, 1)
        # energy: B x S1 x S2

        if self.segmented:
            rel_pos_ids = self.generate_segmented_relative_pos_ids(
                segment_ids,
                self.radius,
                change_direction=self.change_direction,
            ).to(Q.device)
            # rel_pos_ids: S1 x S2
        else:
            rel_pos_ids = self.generate_slided_relative_pos_ids(
                seq_len_q,
                self.radius,
            ).to(Q.device)
            # rel_pos_ids: S1 x S2

        rel_shift = Q_rel.gather(
            2,
            # TODO if seq len won't change, compute this only once
            rel_pos_ids.repeat(batch_size, 1, 1),
        )
        # rel_shift: B x S1 x S2

        energy = (energy + rel_shift) / self.normalizer

        if self.hard_masking:
            # TODO maybe make 0 parametric?
            mask = (rel_pos_ids > 0).repeat(batch_size, 1, 1)

            if padding_mask is not None:
                mask = mask & ~padding_mask

            energy = energy.masked_fill(mask, float("-inf"))

        if (padding_mask is not None) and (not self.hard_masking):
            energy = energy.masked_fill(padding_mask, float("-inf"))


        return energy

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        segment_ids: LongTensor,
        padding_mask: Optional[BoolTensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """

        Args:
            Q (Tensor): B x S1 x d
            K (Tensor): B x S2 x d
            V (Tensor): B x S2 x d
            segment_ids (LongTensor): max(S1, S2)
            padding_mask: (BoolTensor): B x S1 x S2

        Returns:
            Tuple[Tensor, Tensor]:
                attention : B x S1 x S2
                Z         : B x S1 x d
        """

        energy = self.compute_energy(Q, K, segment_ids, padding_mask=padding_mask)

        # compute softmax over S2 dimension
        attn = F.softmax(energy, dim=2)
        # attn: B x S1 x S2

        attn = self.dropout(attn)

        return energy, attn @ V
