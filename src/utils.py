from typing import Tuple
import math

import torch
from torch import Tensor, LongTensor, BoolTensor
import torch.nn.functional as F


def generate_auxiliary_tokens(
    token_ids: LongTensor,
    padding_idx: int = 0,
    long_to_global_ratio: int = 16,
) -> LongTensor:
    """TODO

    Args:
        token_ids (LongTensor): B x Sl
        padding_idx (int, optional): padding index. Defaults to 0.
        long_to_global_ratio (int, optional): TBA. Defaults to 16.

    Returns:
        LongTensor: B x Sg
    """
    batch_size, seq_len = token_ids.shape

    assert seq_len % long_to_global_ratio == 0

    num_global_tokens = seq_len // long_to_global_ratio

    segment_ids = (
        torch.arange(
            0,
            num_global_tokens,
            dtype=token_ids.dtype,
            device=token_ids.device,
        )
        .repeat_interleave(long_to_global_ratio)
        .repeat(batch_size, 1)
    )
    # segment_ids: B x Sl

    global_token_ids = torch.ones(
        (batch_size, num_global_tokens), dtype=token_ids.dtype, device=token_ids.device
    )
    # global_token_ids: B x Sg

    padding_mask = token_ids == padding_idx

    global_padding_mask = padding_mask.reshape(
        batch_size,
        seq_len // long_to_global_ratio,
        long_to_global_ratio,
    ).all(dim=2)
    global_token_ids[global_padding_mask] = padding_idx
    return global_token_ids, segment_ids


def _gen_structural_pos_ids(
    segment_ids: LongTensor,
    long_padding_mask: BoolTensor,
    global_padding_mask: BoolTensor,
    device: str,
    rel_max_dist: int,
    padding_idx: int,
    hard_masking: bool = False,
    transpose: bool = False,
) -> LongTensor:
    batch_size = segment_ids.shape[0]
    long_seq_len = long_padding_mask.shape[1]
    global_seq_len = global_padding_mask.shape[1]

    i = segment_ids.unsqueeze(1).repeat(1, global_seq_len, 1)
    # i: B x Sg x Sl
    j = torch.arange(global_seq_len, device=device).view(1, global_seq_len, 1)
    # j: 1 x Sg x 1
    rel_pos_ids = (i - j).abs()
    # rel_pos_ids: B x Sg x Sl

    structural_mask = torch.zeros(
        (batch_size, global_seq_len, long_seq_len),
        dtype=torch.bool,
        device=device,
    )
    # structural_mask: B x Sg x Sl

    # add masking if position is too far for attention

    if hard_masking:
        structural_mask |= rel_pos_ids > 0

    structural_mask |= global_padding_mask.view(batch_size, global_seq_len, 1)
    structural_mask |= long_padding_mask.view(batch_size, 1, long_seq_len)

    # add 1 to reserve 0 for masking
    rel_pos_ids = rel_pos_ids.clip(max=rel_max_dist) + 1

    if transpose:
        rel_pos_ids = rel_pos_ids.permute(0, 2, 1)
        structural_mask = structural_mask.permute(0, 2, 1)

    # avoid masking all query values, otherwise it will break softmax
    structural_mask = structural_mask & (~structural_mask.all(dim=2, keepdims=True))

    # add masking
    rel_pos_ids[structural_mask] = padding_idx

    return rel_pos_ids


def _gen_sliding_pos_ids(
    padding_mask: BoolTensor,
    device: str,
    rel_max_dist: int,
    padding_idx: int,
    sliding_window_radius: int,
) -> LongTensor:
    batch_size, seq_len = padding_mask.shape

    mask = torch.zeros(
        (batch_size, seq_len, seq_len),
        dtype=torch.bool,
        device=device,
    )
    # mask: B x Sg x Sg

    i = torch.arange(
        seq_len,
        device=device,
    ).repeat(seq_len, 1)
    j = torch.arange(seq_len, device=device).unsqueeze(1)
    rel_pos_ids = (
        (i - j)
        .view(
            1,
            seq_len,
            seq_len,
        )
        .repeat(batch_size, 1, 1)
    )
    # rel_pos_ids: B x Sg x Sg

    # add masking if position is too far for attention
    mask |= rel_pos_ids.abs() > sliding_window_radius

    mask |= padding_mask.view(batch_size, 1, seq_len)

    rel_pos_ids = (
        rel_pos_ids.clip(
            min=-rel_max_dist,
            max=rel_max_dist,
        )
        + rel_max_dist
        + 1
    )  # add 1 to reserve 0 for masking

    # avoid masking all query values, otherwise it will break softmax
    mask = mask & (~mask.all(dim=2, keepdims=True))

    # add masking
    rel_pos_ids[mask] = padding_idx
    return rel_pos_ids


def _gen_fast_sliding_pos_ids(
    padding_mask: BoolTensor,
    device: str,
    rel_max_dist: int,
    padding_idx: int,
    sliding_window_radius: int,
) -> LongTensor:
    batch_size, seq_len = padding_mask.shape

    block_len = sliding_window_radius + 1
    num_blocks = math.ceil(seq_len / block_len)
    BLOCK_SPAN = 3

    mask = torch.zeros(
        (
            batch_size,
            num_blocks,
            block_len,
            BLOCK_SPAN * block_len,
        ),
        dtype=torch.bool,
        device=device,
    )

    pad_left = block_len
    pad_right = block_len + (num_blocks * block_len - seq_len)

    i = torch.arange(0, BLOCK_SPAN * block_len, device=device)
    j = torch.arange(0, block_len, device=device)

    rel_pos_ids = (i - block_len).repeat(block_len, 1) - j.view(block_len, 1)
    mask |= (
        (rel_pos_ids.abs() > sliding_window_radius).view(
            1,
            1,
            block_len,
            BLOCK_SPAN * block_len,
        )
        # mask: 1 x 1 x block_len x (3 * block_len)
    )

    rel_pos_ids = (
        rel_pos_ids.clip(
            min=-rel_max_dist,
            max=rel_max_dist,
        )
        + rel_max_dist
        + 1
    ).repeat(
        batch_size,
        num_blocks,
        1,
        1,
    )
    # rel_pos_ids: B x NB x BL x (3 * BL)

    sliding_ids = (
        torch.arange(0, BLOCK_SPAN * block_len, device=device)
        .view(1, BLOCK_SPAN * block_len)
        .repeat(num_blocks, 1)
    )  # sliding_ids: nbq x (3 * bl)

    sliding_ids += torch.arange(
        0,
        num_blocks * block_len,
        block_len,
        device=device,
    ).view(num_blocks, 1)

    padding_mask = F.pad(
        padding_mask,
        (pad_left, pad_right),
        mode="constant",
        value=True,
    )

    # padding_mask: B x (pad_left + S + pad_right)
    padding_mask = padding_mask[:, sliding_ids].view(
        batch_size,
        num_blocks,
        1,
        BLOCK_SPAN * block_len,
    )
    # padding_mask: B x nbq x 1 x (3 * bl)
    mask |= padding_mask

    # avoid masking all query values, otherwise it will break softmax
    mask = mask & (~mask.all(dim=3, keepdims=True))
    rel_pos_ids[mask] = padding_idx

    # rel_pos_ids: B x NB x BL x (3 * BL)
    return rel_pos_ids


def generate_relative_pos_ids(
    long_token_ids: LongTensor,
    global_token_ids: LongTensor,
    segment_ids: LongTensor,
    padding_idx: int = 0,
    sliding_window_radius: int = 4,
    sliding_rel_max_dist: int = 4,
    structured_rel_max_dist: int = 4,
    g2l_hard_masking: bool = True,
) -> Tuple[LongTensor, LongTensor, LongTensor, LongTensor]:
    """_summary_

    Args:
        long_token_ids (LongTensor): B x Sl
        global_token_ids (LongTensor): B x Sg
        segment_ids (LongTensor): B x Sl
        padding_idx (int, optional): padding index. Defaults to 0.
        # TODO

    Returns:
        Tuple[LongTensor, LongTensor, LongTensor, LongTensor]:
            LongTensor: l2l_rel_pos_ids as B x NB x BL x (3 * BL)
            LongTensor: l2g_rel_pos_ids as B x Sl x Sg
            LongTensor: g2g_rel_pos_ids as B x Sg x Sg
            LongTensor: g2l_rel_pos_ids as B x Sg x Sl
    """

    device = long_token_ids.device
    long_padding_mask = long_token_ids == padding_idx
    # long_padding_mask: B x Sl
    global_padding_mask = global_token_ids == padding_idx
    # global_padding_mask: B x Sg

    l2l_rel_pos_ids = _gen_fast_sliding_pos_ids(
        long_padding_mask,
        device,
        sliding_rel_max_dist,
        padding_idx,
        sliding_window_radius,
    )
    # l2l_rel_pos_ids: B x NB x BL x (3 * BL)

    l2g_rel_pos_ids = _gen_structural_pos_ids(
        segment_ids,
        long_padding_mask,
        global_padding_mask,
        device,
        structured_rel_max_dist,
        padding_idx,
        transpose=True,
    )
    # l2g_rel_pos_ids: B x Sl x Sg

    g2g_rel_pos_ids = _gen_sliding_pos_ids(
        global_padding_mask,
        device,
        sliding_rel_max_dist,
        padding_idx,
        math.inf,
    )
    # g2g_rel_pos_ids: B x Sg x Sg

    g2l_rel_pos_ids = _gen_structural_pos_ids(
        segment_ids,
        long_padding_mask,
        global_padding_mask,
        device,
        structured_rel_max_dist,
        padding_idx,
        hard_masking=g2l_hard_masking,
    )
    # g2l_rel_pos_ids: B x Sg x Sl

    return l2l_rel_pos_ids, l2g_rel_pos_ids, g2g_rel_pos_ids, g2l_rel_pos_ids


def seq_to_blocks(x: Tensor, block_len: int) -> Tensor:
    """TODO

    Args:
        x (Tensor): B' x S x d'

    Returns:
        Tensor: B' x NB x BL x d'
    """
    batch_size, seq_len, d = x.shape

    num_blocks = math.ceil(seq_len / block_len)

    pad_size = num_blocks * block_len - seq_len

    return F.pad(
        x,
        (0, 0, 0, pad_size),
        mode="constant",
        value=0,
    ).view(
        batch_size,
        num_blocks,
        block_len,
        d,
    )


def blocks_to_seq(x: Tensor, seq_len: int) -> Tensor:
    """TODO

    Args:
        x (Tensor): B' x NB x BL x d'

    Returns:
        Tensor: B' x S x d'
    """
    return x.flatten(
        start_dim=1,
        end_dim=2,
    )[:, :seq_len, :]


def blocks_to_grouped_blocks(x: Tensor) -> Tensor:
    """TODO

    Args:
        x (Tensor): B' x NB x BL x d'

    Returns:
        Tensor: B' x NB x (3 * BL) x d'
    """
    num_blocks, block_len = x.shape[1:3]

    sliding_ids = (
        torch.arange(0, 3 * block_len, device=x.device)
        .view(1, 3 * block_len)
        .repeat(num_blocks, 1)
    )  # sliding_ids: NB x (3 * BL)

    sliding_ids += torch.arange(
        0,
        num_blocks * block_len,
        block_len,
        device=x.device,
    ).view(num_blocks, 1)

    return F.pad(
        x.flatten(start_dim=1, end_dim=2),
        (0, 0, block_len, block_len),
        mode="constant",
        value=0,
    )[:, sliding_ids, :]
