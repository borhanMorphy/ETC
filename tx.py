import torch
from src.blocks.attention import FastRelativeMHA, RelativeMultiHeadAttention


SEQ_LEN = 16
batch_size = 2

embed_dim = 16
num_heads = 2

rel_pos_max_distance = 2
local_attention_radius = 2

fast_model = FastRelativeMHA(
    embed_dim,
    num_heads,
    rel_pos_max_distance,
    local_attention_radius=local_attention_radius,
    kdim=embed_dim//2,
    vdim=embed_dim//2,
    skip_query_projection=True,
)

model = RelativeMultiHeadAttention(
    embed_dim,
    num_heads,
    rel_pos_max_distance,
    local_attention_radius=local_attention_radius,
    kdim=embed_dim//2,
    vdim=embed_dim//2,
    skip_query_projection=True,
)

# B x S x d
Q = torch.rand(batch_size, SEQ_LEN, embed_dim)
K = torch.rand(batch_size, SEQ_LEN, embed_dim//2)
V = torch.rand(batch_size, SEQ_LEN, embed_dim//2)
segment_ids = torch.randint(0, 2, size=(batch_size, SEQ_LEN))
key_padding_mask = torch.zeros((batch_size, SEQ_LEN), dtype=torch.bool)
#key_padding_mask[:, 10:] = True

model.load_state_dict(fast_model.state_dict())

model.eval()
fast_model.eval()

fast_out = fast_model.forward(
    Q,
    K,
    V,
    segment_ids,
    key_padding_mask=key_padding_mask,
)

out = model.forward(
    Q,
    K,
    V,
    segment_ids,
    key_padding_mask=key_padding_mask,
)
# B x S x d
print(torch.isclose(fast_out, out))
