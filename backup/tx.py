import torch
from src import ETC, ModelConfig

import torch.nn as nn

from dataclasses import dataclass


def main():
    BATCH_SIZE = 3
    SEQ_LEN = 1024

    model_config = ModelConfig(
        sliding_window_radius=7,
        segment_radius=1, # this enabled to assign different pos embeddings for further global tokens
        num_features=512,
        num_heads=16,
        vocab_size=512,
        hard_masking=True,
        global_token_ratio=16, # how "long tokens" will be assigned to single global token
    )

    model = ETC(
        model_config.num_features,
        model_config.num_heads,
        model_config.vocab_size,
        model_config.sliding_window_radius,
        model_config.segment_radius,
        hard_masking=model_config.hard_masking,
        global_token_ratio=model_config.global_token_ratio,
        num_of_layers=2,
    )
    print(model)

    print(model_config)

    token_ids = torch.randint(low=0, high=model_config.vocab_size, size=(BATCH_SIZE, SEQ_LEN))
    print("input -> ", token_ids.shape)

    logits = model(token_ids)

    print("output -> ", logits.shape)


if __name__ == "__main__":
    main()
