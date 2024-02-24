from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor, LongTensor, BoolTensor


class FixedRatioGlobalBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        long_to_global_ratio: int = 16,
        add_cls_token: bool = False,
    ):
        super().__init__()

        self.long_to_global_ratio = long_to_global_ratio
        extra_num_tokens = 2 if add_cls_token else 1
        self.embeds = nn.Embedding(
            1 + extra_num_tokens,
            hidden_size,
            padding_idx=0,
        )
        self.add_cls_token = add_cls_token


    def forward(self, token_ids: LongTensor, padding_mask: BoolTensor = None) -> Tuple[Tensor, BoolTensor]:
        """_summary_

        Args:
            token_ids (LongTensor): B x Sl
            padding_mask (BoolTensor, optional): B x Sl. Defaults to None.

        Returns:
            Tuple[Tensor, BoolTensor]:
                Tensor: B x Sg x d
                BoolTensor: B x Sg
        """

        batch_size, seq_len = token_ids.shape
        global_padding_mask = None

        assert seq_len % self.long_to_global_ratio == 0

        num_global_tokens = seq_len // self.long_to_global_ratio

        global_token_ids = torch.ones((batch_size, num_global_tokens), dtype=token_ids.dtype, device=token_ids.device)

        if padding_mask is not None:
            B, Sl = padding_mask.shape
            global_padding_mask = padding_mask.reshape(
                B, Sl // self.long_to_global_ratio, self.long_to_global_ratio,
            ).all(dim=2)
            global_token_ids[global_padding_mask] = self.embeds.padding_idx

        return self.embeds(global_token_ids), global_padding_mask
