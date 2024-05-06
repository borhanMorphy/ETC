from src import ETC, config

import torch

BATCH_SIZE = 2
PAD_SIZE = 1
SEQ_SIZE = 63

model_config = config.ModelConfig(
    d_model=128,
    num_heads=4,
    dim_feedforward=2 * 128,
    dropout=0.0,
    num_layers=1,
    num_classes=-1,  # TODO
    vocab_size=50,
    padding_idx=0,
    long_to_global_ratio=16,
    l2l=config.ETCAttentionConfig(
        rel_pos_max_distance=16,
        local_attention_radius=5,
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
        rel_pos_max_distance=3,
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

token_ids = torch.randint(1, 50, (BATCH_SIZE, SEQ_SIZE + PAD_SIZE))

if PAD_SIZE > 0:
    token_ids[:, -PAD_SIZE:] = model_config.padding_idx


model = ETC(model_config)


print(model)

long_outs, global_outs = model(token_ids)

print("long_outs -> ", long_outs.shape)
print("global_outs -> ", global_outs.shape)
