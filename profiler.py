from torch.profiler import profile, record_function, ProfilerActivity

from src import VanillaTransformer, ETC, config

import torch


model_config = config.ModelConfig(
    d_model=128,
    num_heads=4,
    dim_feedforward=2 * 128,
    dropout=0.0,
    num_layers=2,
    num_classes=-1,  # TODO
    vocab_size=50,
    padding_idx=0,
    long_to_global_ratio=64,
    l2l=config.ETCAttentionConfig(
        rel_pos_max_distance=64,
        local_attention_radius=64,
        attention_type="sparse",
        directed_relative_position=True,
    ),
    l2g=config.ETCAttentionConfig(
        rel_pos_max_distance=8,
        local_attention_radius=None,
        attention_type="dense",
        directed_relative_position=False,
    ),
    g2g=config.ETCAttentionConfig(
        rel_pos_max_distance=8,
        local_attention_radius=None,
        attention_type="dense",
        directed_relative_position=True,
    ),
    g2l=config.ETCAttentionConfig(
        rel_pos_max_distance=8,
        local_attention_radius=None,
        attention_type="dense",
        directed_relative_position=False,
    ),
)

SEQ_LEN = 2**14

inputs = torch.randint(low=0, high=model_config.vocab_size, size=(1, SEQ_LEN))


# model = VanillaTransformer(
#     model_config.d_model,
#     model_config.num_heads,
#     model_config.vocab_size,
#     SEQ_LEN,
#     num_layers=model_config.num_layers,
# )


model = ETC(model_config)

# with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
#    with record_function("model_inference"):
#        model(inputs)

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

with profile(
    activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
