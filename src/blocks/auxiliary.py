import torch
import torch.nn as nn
from torch import LongTensor


class FixedRatioGlobalBlock(nn.Module):
    def __init__(self, long_to_global_ratio: int = 16):
        super().__init__()
        self.long_to_global_ratio = long_to_global_ratio


    def forward(self, token_ids: LongTensor) -> LongTensor:
        """Generates global token ids

        Args:
            token_ids (LongTensor): B x Sl

        Returns:
            LongTensor: B x Sl
        """

        batch_size, seq_len = token_ids.shape
        global_padding_mask = None

        assert seq_len % self.long_to_global_ratio == 0

        num_global_tokens = seq_len // self.long_to_global_ratio

        global_token_ids = torch.ones((batch_size, num_global_tokens), dtype=token_ids.dtype, device=token_ids.device)
        segment_ids = (
            torch.arange(
                0,
                num_global_tokens,
                dtype=token_ids.dtype,
                device=token_ids.device,
            )
            .repeat_interleave(self.long_to_global_ratio)
            .repeat(batch_size, 1)
        )
        # segment_ids: B x Sl
        return segment_ids
