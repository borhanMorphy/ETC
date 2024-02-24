from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_features: int
    num_heads: int
    vocab_size: int

    long_to_global_ratio: int
    add_global_cls_token: bool
    rel_pos_max_distance: int
    local_attention_radius: int