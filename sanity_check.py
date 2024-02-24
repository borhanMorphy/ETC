from random import randint

import matplotlib.pyplot as plt
import numpy as np
import torch
from src import ETC, ModelConfig

import torch.nn as nn


def get_rand_color():
    r = randint(0, 254)
    g = randint(0, 254)
    b = randint(0, 254)
    return (r, g, b)


RESERVED_COLORS = {
    float("-inf"): (255, 255, 255)
}

def viz_img(attn):
    if len(attn.shape) == 3:
        attn = attn[0]
    un_vals, ids = torch.unique(attn, return_inverse=True)

    id2color = {}

    for idx, val in enumerate(un_vals):
        val = val.item()
        if val in RESERVED_COLORS:
            id2color[idx] = RESERVED_COLORS[val]
        else:
            id2color[idx] = get_rand_color()

    h, w = ids.shape

    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img[i, j] = id2color[ids[i, j].item()]

    plt.imshow(img)
    plt.show()


def main():
    BATCH_SIZE = 1

    model_config = ModelConfig(
        global_seq_len=3,
        long_seq_len=8,
        sliding_window_radius=2,
        segment_radius=1,
        num_features=32,
        num_heads=1,
        hard_masking=True,
    )
    Q_global = torch.ones((BATCH_SIZE, model_config.global_seq_len, model_config.num_features // (2 * model_config.num_heads)))
    K_global = torch.zeros((BATCH_SIZE, model_config.global_seq_len, model_config.num_features // (2 * model_config.num_heads)))

    Q_long = torch.ones((BATCH_SIZE, model_config.long_seq_len, model_config.num_features // (2 * model_config.num_heads)))
    K_long = torch.zeros((BATCH_SIZE, model_config.long_seq_len, model_config.num_features // (2 * model_config.num_heads)))

    segment_ids = torch.LongTensor([0, 0, 0, 1, 1, 2, 2, 2])

    model = ETC(
        model_config.num_features,
        model_config.num_heads,
        model_config.sliding_window_radius,
        model_config.segment_radius,
        hard_masking=model_config.hard_masking,
    )
    print(model)

    g2g_energy = model.global_layer.g2g_attention.rel_attention_layers[
        0
    ].compute_energy(Q_global, K_global, segment_ids)

    g2l_energy = model.global_layer.g2l_attention.rel_attention_layers[
        0
    ].compute_energy(Q_global, K_long, segment_ids)

    l2g_energy = model.local_layer.l2g_attention.rel_attention_layers[
        0
    ].compute_energy(Q_long, K_global, segment_ids)


    viz_img(g2g_energy)
    viz_img(g2l_energy)

    viz_img(l2g_energy)


if __name__ == "__main__":
    main()
