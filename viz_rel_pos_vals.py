import torch
import matplotlib.pyplot as plt
import numpy as np

from src import utils


def viz_rel_pos_ids(rel_pos_ids, color_type: str, rel_max_dist: int, padding_idx: int):
    int_to_color = {
        "structural": {
            "pad": (255, 255, 255),  # <padding> white
            0: (0, 0, 0),  # 0         black
            1: (245, 39, 39),  # 1    red
            2: (32, 215, 12),  # 2    green
            3: (63, 26, 212),  # 3    blue
            4: (120, 120, 120),  # 4  gray
        },
        "sliding": {
            "pad": (255, 255, 255),  # <padding> white
            -4: (120, 120, 120),  # -4  gray
            -3: (63, 26, 212),  # -3    blue
            -2: (32, 215, 12),  # -2    green
            -1: (245, 39, 39),  # -1    red
            0: (0, 0, 0),  # 0   black
            1: (211, 150, 150),  # +1  soft red
            2: (124, 226, 126),  # +2  soft green
            3: (153, 141, 229),  # +3  soft blue
            4: (180, 180, 180),  # +4  soft gray
        },
    }

    _, h, w = rel_pos_ids.shape

    c_attention_viz = np.zeros((h, w, 3), dtype=np.uint8)

    lower_bound = rel_pos_ids.min()
    upper_bound = rel_pos_ids.max()

    for i in range(lower_bound, upper_bound + 1):
        if i == padding_idx:
            key = "pad"
        elif color_type == "sliding":
            key = i - 1 - rel_max_dist
        elif color_type == "structural":
            key = i - 1
        else:
            assert False
        mask = (rel_pos_ids.squeeze(0) == i).numpy()
        c_attention_viz[mask, :] = int_to_color[color_type][key]

    plt.imshow(c_attention_viz)
    plt.show()


B = 1
S = 63
P = 1
structured_rel_max_dist = 1
sliding_rel_max_dist = 2
sliding_window_radius = 2
padding_idx = 0
g2l_hard_masking = True


token_ids = torch.randint(1, 50, (B, S + P))

if P > 0:
    token_ids[:, -P:] = padding_idx
print("long tokens -> ", token_ids)

g_token_ids, s_ids = utils.generate_auxiliary_tokens(
    token_ids,
    padding_idx=padding_idx,
    long_to_global_ratio=4,
)

print("global tokens -> ", g_token_ids)
print("segment ids -> ", s_ids)

(
    l2l_rel_pos_ids,
    l2g_rel_pos_ids,
    g2g_rel_pos_ids,
    g2l_rel_pos_ids,
) = utils.generate_relative_pos_ids(
    token_ids,
    g_token_ids,
    s_ids,
    padding_idx=padding_idx,
    sliding_rel_max_dist=sliding_rel_max_dist,
    sliding_window_radius=sliding_window_radius,
    structured_rel_max_dist=structured_rel_max_dist,
    g2l_hard_masking=g2l_hard_masking,
)

viz_rel_pos_ids(l2g_rel_pos_ids, "structural", structured_rel_max_dist, padding_idx)
viz_rel_pos_ids(g2l_rel_pos_ids, "structural", structured_rel_max_dist, padding_idx)
viz_rel_pos_ids(g2g_rel_pos_ids, "sliding", sliding_rel_max_dist, padding_idx)
print(l2l_rel_pos_ids.shape)
for i in range(l2l_rel_pos_ids.shape[1]):
    viz_rel_pos_ids(
        l2l_rel_pos_ids[:, i, :, :], "sliding", sliding_rel_max_dist, padding_idx
    )
