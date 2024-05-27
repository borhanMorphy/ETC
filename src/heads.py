from typing import Dict

import torch.nn as nn
from torch import Tensor, LongTensor

from .module import ETC
from .config import ModelConfig


class ETCPreTraining(ETC):
    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        self.mlm_head = nn.Linear(
            model_config.d_model,
            model_config.vocab_size,
            bias=False,
        )

    def compute_loss(
        self,
        input_tokens_ids: Dict[str, LongTensor],
        # TODO add targets
        mlm_weight: float = 0.8,
        cpc_weight: float = 0.2,
    ) -> Tensor:
        """_summary_

        Args:
            # TODO
            long_token_ids (Tensor): B x Sl
            global_token_ids (Tensor): B x Sg
            segment_ids (Tensor): B x Sl

        Returns:
            Tuple[Tensor, Tensor]:
                Tensor: B x Sl x d
                Tensor: B x Sg x d
        """
        z_long, z_global = self.forward(
            input_tokens_ids["long_token_ids"],
            input_tokens_ids["global_token_ids"],
            input_tokens_ids["segment_ids"],
        )

        # compute logits for long

        mlm_logits = self.mlm_head(z_long)

        return mlm_logits
