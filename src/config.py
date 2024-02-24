from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_features: int
    num_heads: int
    vocab_size: int

    sliding_window_radius: int
    segment_radius: int
    hard_masking: bool = False
    global_token_ratio: int = 16
