from typing import Optional

import torch
from torch import Tensor, LongTensor, BoolTensor
import torch.nn as nn


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
