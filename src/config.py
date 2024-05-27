from typing import Optional, Literal
from dataclasses import dataclass, field


@dataclass
class MLMConfig:
    mask_perc: float = 0.15
    replace_with_mask_token_perc = 0.80
    replace_with_random_token_perc: float = 0.10
    keep_same_token_perc: float = 0.10

    # sanity checks
    def __post_init__(self):
        assert (
            0.0 <= self.mask_perc <= 1.0
        ), "mask percantage can not be larger than 1 and lower than 0"
        # TODO add more


@dataclass
class CPCConfig:
    sentence_mask_perc: float = 0.10

    # sanity checks
    def __post_init__(self):
        assert (
            0.0 <= self.sentence_mask_perc <= 1.0
        ), "sentence mask percantage can not be larger than 1 and lower than 0"


@dataclass
class PreTrainingConfig:
    mlm_loss_weight: float = 0.8
    cpc_loss_weight: float = 0.2

    mlm_config: MLMConfig = field(default_factory=MLMConfig)
    cpc_config: CPCConfig = field(default_factory=CPCConfig)

    @property
    def is_cpc_enabled(self) -> bool:
        return self.cpc_loss_weight != 0.0

    # sanity checks
    def __post_init__(self):
        assert (
            self.mlm_loss_weight + self.cpc_loss_weight
        ) == 1.0, "sum of MLM and CPC must be eqaul to 1"


@dataclass
class ETCAttentionConfig:
    rel_pos_max_distance: int
    local_attention_radius: Optional[int]
    attention_type: Literal["dense", "sparse"]
    directed_relative_position: bool


@dataclass
class TransformerLayerConfig:
    d_model: int
    num_heads: int
    dim_feedforward: int
    dropout: float

    # sanity checks
    def __post_init__(self):
        assert (
            0.0 <= self.dropout <= 0.5
        ), "dropout should be in range of [0, 0.5), larger values will cause problems and cannot be negative"
        assert self.d_model > 0
        assert self.d_model % 2 == 0
        assert self.num_heads > 0
        assert self.dim_feedforward > 0
        assert self.dim_feedforward % 2 == 0

        assert (
            self.d_model % self.num_heads == 0
        ), "`d_model` needs to be divisible by `self.num_heads`"
        assert (
            self.d_model <= self.dim_feedforward
        ), "`dim_feedforward` needs to be larger than `d_model` or at least equal."


@dataclass
class ModelConfig(TransformerLayerConfig):
    num_layers: int

    # including global token types + padding
    vocab_size: int
    padding_idx: int

    long_to_global_ratio: int

    l2l: ETCAttentionConfig
    l2g: ETCAttentionConfig
    g2g: ETCAttentionConfig
    g2l: ETCAttentionConfig
