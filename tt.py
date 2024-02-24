import torch
import torch.nn as nn

from src.blocks import SlidingWindowAttention


max_seq_len = 2000
attention_radius = 100
max_rel_distance = 5
batch_size = 1
hidden = 16
num_heads = 1

sw_attn = SlidingWindowAttention(
    hidden,
    max_seq_len,
    attention_radius,
    max_rel_distance,
)

mh_attn = nn.MultiheadAttention(
    hidden,
    num_heads=num_heads,
    batch_first=True,
)


import time
import matplotlib.pylab as plt


def run_test(fast=False):
    exps = list()

    for seq_len in range(100, 2000):
        q = torch.rand(batch_size, seq_len, hidden)
        k = torch.rand(batch_size, seq_len, hidden)
        v = torch.rand(batch_size, seq_len, hidden)

        start = time.time()
        if fast:
            v_hat, attn = sw_attn(q, k, v)
        else:
            v_hat, attn = mh_attn(q, k, v)
        end = time.time()
        exps.append((seq_len, end - start))

    return list(zip(*exps))


plt.ylim(0, 0.02)

seq_lens, elapsed_times = run_test(fast=False)
plt.plot(seq_lens, elapsed_times)
seq_lens, elapsed_times = run_test(fast=True)
plt.plot(seq_lens, elapsed_times)
plt.show()
