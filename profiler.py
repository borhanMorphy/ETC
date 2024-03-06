from torch.profiler import profile, record_function, ProfilerActivity

from src import VanillaTransformer, ETC, ModelConfig, ETCAttentionConfig

import torch



config = ModelConfig(
    num_layers=2,
    num_classes=1,

    vocab_size=32, #tokenizer.vocab_size,
    padding_idx=0, #tokenizer.padding_idx,

    d_model=128,
    num_heads=4,
    dim_feedforward=128*2,
    dropout=0.0,

    long_to_global_ratio=16,
    add_global_cls_token=False,

    l2l=ETCAttentionConfig(
        rel_pos_max_distance=5,
        local_attention_radius=5*5,
        attention_type="slided",
    ),
    l2g=ETCAttentionConfig(
        rel_pos_max_distance=1,
        local_attention_radius=None,
        attention_type="segmented",
    ),
    g2g=ETCAttentionConfig(
        rel_pos_max_distance=2,
        local_attention_radius=None,
        attention_type="slided",
    ),
    g2l=ETCAttentionConfig(
        rel_pos_max_distance=2,
        local_attention_radius=2,
        attention_type="segmented",
    ),
)

SEQ_LEN = 2**12

inputs = torch.randint(low=0, high=config.vocab_size, size=(1, SEQ_LEN))

#model = VanillaTransformer(
#    config.d_model,
#    config.num_heads,
#    config.vocab_size,
#    SEQ_LEN,
#    num_classes=config.num_classes,
#    num_layers=config.num_layers,
#)
model = ETC(config)

#with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#    with record_function("model_inference"):
#        model(inputs)
#
#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

with profile(activities=[ProfilerActivity.CPU],
        profile_memory=True, record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))