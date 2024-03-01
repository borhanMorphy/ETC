from typing import Optional, Literal
from dataclasses import dataclass, field

@dataclass
class ETCAttentionConfig:
    rel_pos_max_distance: int
    local_attention_radius: Optional[int]
    attention_type: Literal["slided", "segmented"]


@dataclass
class TransformerLayerConfig:
    d_model: int
    num_heads: int
    dim_feedforward: int
    dropout: float

    # sanity checks
    def __post_init__(self):
        assert 0.0 <= self.dropout <= 0.5, "dropout should be in range of [0, 0.5), larger values will cause problems and cannot be negative"
        assert self.d_model > 0
        assert self.d_model % 2 == 0
        assert self.num_heads > 0
        assert self.dim_feedforward > 0
        assert self.dim_feedforward % 2 == 0

        assert self.d_model % self.num_heads == 0, "`d_model` needs to be divisible by `self.num_heads`"
        assert self.d_model <= self.dim_feedforward, "`dim_feedforward` needs to be larger than `d_model` or at least equal."



@dataclass
class ModelConfig(TransformerLayerConfig):
    num_layers: int
    num_classes: int

    vocab_size: int
    padding_idx: int

    long_to_global_ratio: int
    add_global_cls_token: bool

    l2l: ETCAttentionConfig
    l2g: ETCAttentionConfig
    g2g: ETCAttentionConfig
    g2l: ETCAttentionConfig