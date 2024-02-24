from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor, BoolTensor

from .relative_position import RelativePE


class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        radius: int,
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_k = d_out // num_heads
        self.num_heads = num_heads
        # r
        self.radius = radius
        # R
        self.window_size = 2 * radius + 1

        self.rel_embeds = nn.ModuleList([
            nn.Embedding(
                self.window_size,
                self.d_k,
            )
            for _ in range(num_heads)
        ])
        self.padding = nn.ConstantPad2d(
            (0, 0, radius, radius),
            value=0,
        )

        self.query_linear = nn.Linear(d_in, d_out, bias=False)
        self.key_linear = nn.Linear(d_in, d_out, bias=False)
        self.value_linear = nn.Linear(d_in, d_out, bias=False)
        self.combine_linear = nn.Linear(d_out, d_out, bias=False)

        self.register_buffer(
            "normalizer",
            torch.tensor(self.d_k**0.5),
        )

        self.register_buffer(
            "rel_pos_ids",
            torch.arange(self.window_size),
        )


    @staticmethod
    def generate_sliding_window_ids(
        window_size: int, seq_len: int
    ) -> LongTensor:
        """_summary_

        Args:
            window_size (int): _description_
            seq_len (int): _description_

        Returns:
            LongTensor: S x R
        """
        ids = (
            torch.arange(window_size)
            .repeat(
                seq_len,
            )
            .reshape(
                seq_len,
                window_size,
            )
        )

        ids += torch.arange(seq_len).reshape(seq_len, 1)
        return ids

    @staticmethod
    def generate_attention_mask_ids(
        radius: int,
        seq_len: int,
    ) -> LongTensor:
        i_upper, j_upper = torch.triu_indices(
            radius, radius
        )
        # get mirror to y axis
        j_upper = (radius - 1) - j_upper

        i_lower, j_lower = torch.tril_indices(
            radius, radius
        )
        # get mirror to y axis
        j_lower = (radius - 1) - j_lower

        # shift x axis
        j_lower += radius + 1

        # shift y axis
        i_lower += seq_len - radius

        i = torch.cat([i_upper, i_lower])
        j = torch.cat([j_upper, j_lower])

        return torch.stack([i, j])

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        padding_mask: BoolTensor = None,
    ) -> Tensor:
        """

        Args:
            Q (Tensor): B x S x d_out
            K (Tensor): B x S x d_in
            V (Tensor): B x S x d_in
            padding_mask (BoolTensor): B x S

        Returns:
            Tensor: B x S x d_out
        """
        batch_size, seq_len = Q.shape[:2]

        K = (
            self.key_linear(K)
            .view(
                batch_size,
                seq_len,
                self.d_k,
                self.num_heads,
            )
            .permute(0, 3, 1, 2)  # -> B x h x S x d_k
        )
        V = (
            self.value_linear(V)
            .view(
                batch_size,
                seq_len,
                self.d_k,
                self.num_heads,
            )
            .permute(0, 3, 1, 2)  # -> B x h x S x d_k
        )
        Q = self.query_linear(Q).view(
            batch_size,
            seq_len,
            self.d_k,
            self.num_heads,
        ).permute(
            0, 3, 1, 2
        )  # -> B x h x S x d_k

        # TODO avoid loop here
        head_results = []
        for i in range(self.num_heads):
            x = self.single_head_forward(
                Q[:, i, :, :],
                K[:, i, :, :],
                V[:, i, :, :],
                self.rel_embeds[i](self.rel_pos_ids),
                padding_mask=padding_mask,
            )
            # x: B x S x d_k
            head_results.append(x)

        head_results = (
            torch.stack(head_results, dim=1)
            .permute(0, 2, 3, 1)
            .reshape(batch_size, seq_len, self.d_k * self.num_heads)
        )

        return self.combine_linear(head_results)

    def single_head_forward(self, Q, K, V, E, padding_mask = None):
        """
        Args:
            Q (Tensor): B x S x d_k
            K (Tensor): B x S x d_k
            V (Tensor): B x S x d_k
            E (Tensor): R x d_k
            padding_mask (BoolTensor): B x S

        Returns:
            Tensor: B x S x d_k
        """
        # TODO handle what if padding_mask missing

        batch_size, seq_len = Q.shape[:2]

        K_padded = self.padding(K)
        # K_padded: B x (r+S+r) x d_k

        padding_mask_shifted = self.padding(padding_mask.unsqueeze(2)).squeeze(2)
        # padding_mask_shifted: B x (r+S+r)

        sliding_window_ids = self.generate_sliding_window_ids(
            self.window_size,
            seq_len,
        )
        # sliding_window_ids: S x R

        K_hat = K_padded[:, sliding_window_ids].flatten(
            start_dim=0,
            end_dim=1,
        ) #  B x S x d_k -> B x S x R x d_k -> (B * S) x R x d_k

        # add positional embeddings
        K_hat = K_hat + E

        energy = torch.bmm(
            Q.view(batch_size * seq_len, 1, self.d_k), # B x S x d_k -> (B * S) x 1 x d_k
            K_hat.permute(0, 2, 1), # (B * S) x R x d_k -> (B * S) x d_k x R
        ).reshape(
            batch_size,
            seq_len,
            self.window_size,
        ) # (B * S) x 1 x R -> B x S x R

        energy = energy / self.normalizer

        # mask before softmax
        i, j = self.generate_attention_mask_ids(
            self.radius,
            seq_len,
        )
        energy[:, i, j] = float("-inf")
        m = energy == float("-inf")

        mask = padding_mask_shifted[:, sliding_window_ids] | m

        m2 = padding_mask_shifted[:, sliding_window_ids] & (~mask.all(dim=2, keepdims=True))
        
        energy = energy.masked_fill(m2, float("-inf"))

        import matplotlib.pyplot as plt
        #plt.imshow(m2[0].cpu())
        #plt.show()
        #plt.imshow(mask.all(dim=2, keepdims=True)[0].cpu())
        #plt.show()
        #plt.imshow(padding_mask_shifted[:, sliding_window_ids][0].cpu())
        #plt.show()
        #plt.imshow(m[0].cpu())
        #plt.show()


        m = energy == float("-inf")
        plt.imshow(m[0].cpu())
        plt.show()

        # compute softmax over window dimension
        attn = F.softmax(energy, dim=2)
        # attn: B x S x R
        plt.imshow(attn[0].detach().cpu())
        plt.show()

        V_padded = self.padding(V)
        # V_padded: B x (r+S+r) x d_k

        V_hat = V_padded[:, sliding_window_ids]
        # V_hat: B x S x R x d_k

        # apply the attention via element-wise multiplication and aggregation
        V_hat = (attn.unsqueeze(3) * V_hat).sum(dim=2)
        # V_hat: B x S x d_k
        return V_hat


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        radius: int,
        segmented: bool = False,
        change_direction: bool = False,
        hard_masking: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_k = d_out // num_heads
        self.num_heads = num_heads

        self.key_linear = nn.Linear(d_in, d_out, bias=False)
        self.value_linear = nn.Linear(d_in, d_out, bias=False)
        self.combine_linear = nn.Linear(d_out, d_out, bias=False)

        self.rel_attention_layers = nn.ModuleList(
            [
                RelativePE(
                    self.d_k,
                    radius,
                    segmented=segmented,
                    change_direction=change_direction,
                    hard_masking=hard_masking,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        segment_ids: LongTensor,
        padding_mask: BoolTensor = None,
    ) -> Tensor:
        """

        Args:
            Q (Tensor): B x S1 x d_out
            K (Tensor): B x S2 x d_in
            V (Tensor): B x S2 x d_in
            segment_ids (LongTensor): max(S1, S2)
            padding_mask (BoolTensor): B x S1 x S2

        Returns:
            Tensor: B x S1 x d_out
        """
        batch_size, seq_len_q = Q.shape[:2]
        seq_len_k = K.shape[1]
        K = (
            self.key_linear(K)
            .view(
                batch_size,
                seq_len_k,
                self.d_k,
                self.num_heads,
            )
            .permute(0, 3, 1, 2)  # -> B x h x S2 x d_k
        )
        V = (
            self.value_linear(V)
            .view(
                batch_size,
                seq_len_k,
                self.d_k,
                self.num_heads,
            )
            .permute(0, 3, 1, 2)  # -> B x h x S2 x d_k
        )
        Q_proj = Q.view(
            batch_size,
            seq_len_q,
            self.d_k,
            self.num_heads,
        ).permute(
            0, 3, 1, 2
        )  # -> B x h x S1 x d_k

        # TODO avoid loop here
        head_results = []
        for i, rel_attn_layer in enumerate(self.rel_attention_layers):
            _, x = rel_attn_layer(
                Q_proj[:, i, :, :],
                K[:, i, :, :],
                V[:, i, :, :],
                segment_ids,
                padding_mask=padding_mask,
            )
            # x: B x S1 x d_k
            head_results.append(x)

        head_results = (
            torch.stack(head_results, dim=1)
            .permute(0, 2, 3, 1)
            .reshape(batch_size, seq_len_q, self.d_k * self.num_heads)
        )

        return self.combine_linear(head_results)


class GlobalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sliding_window_radius: int,
        segment_radius: int,
        hard_masking: bool = False,
    ):
        super().__init__()
        self.g2g_attention = MultiHeadAttention(
            d_model,
            d_model // 2,
            num_heads,
            sliding_window_radius,
            segmented=False,
        )
        self.g2l_attention = MultiHeadAttention(
            d_model,
            d_model // 2,
            num_heads,
            segment_radius,
            segmented=True,
            hard_masking=hard_masking,
        )
        self.query_projection = nn.Linear(d_model, d_model // 2, bias=False)

    def forward(
        self, x_long: Tensor, x_global: Tensor, segment_ids: LongTensor,
        g2g_padding_mask: BoolTensor = None,
        g2l_padding_mask: BoolTensor = None,
    ) -> Tensor:
        """

        Args:
            x_long (Tensor): B x Sl x d
            x_global (Tensor): B x Sg x d
            segment_ids (LongTensor): Sl
            g2g_padding_mask (BoolTensor): B x Sg x Sg
            g2l_padding_mask (BoolTensor): B x Sg x Sl

        Returns:
            Tensor: B x Sg x d
        """
        Q = self.query_projection(x_global)
        # Q: B x Sg x d/2

        g2g_z = self.g2g_attention(Q, x_global, x_global, segment_ids, padding_mask=g2g_padding_mask)
        # g2g_z: B x Sg x d/2
        g2l_z = self.g2l_attention(Q, x_long, x_long, segment_ids, padding_mask=g2l_padding_mask)
        # g2l_z: B x Sg x d/2

        return torch.cat([g2g_z, g2l_z], dim=2)  # -> B x Sg x d


class LocalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sliding_window_radius: int,
        segment_radius: int,
    ):
        super().__init__()
        self.l2l_attention = SlidingWindowAttention(
            d_model,
            d_model // 2,
            num_heads,
            sliding_window_radius,
        )
        self.l2g_attention = MultiHeadAttention(
            d_model,
            d_model // 2,
            num_heads,
            segment_radius,
            segmented=True,
            change_direction=True,
        )
        self.query_projection = nn.Linear(d_model, d_model // 2, bias=False)

    def forward(
        self, x_long: Tensor, x_global: Tensor, segment_ids: LongTensor,
        l2l_padding_mask: BoolTensor = None,
        l2g_padding_mask: BoolTensor = None,
    ) -> Tensor:
        """

        Args:
            x_long (Tensor): B x Sl x d
            x_global (Tensor): B x Sg x d
            segment_ids (LongTensor): Sl
            l2l_padding_mask (BoolTensor): B x Sl
            l2g_padding_mask (BoolTensor): B x Sl x Sg

        Returns:
            Tensor: B x Sl x d
        """
        Q = self.query_projection(x_long)
        # Q: B x Sl x d/2


        l2l_z = self.l2l_attention(Q, x_long, x_long, padding_mask=l2l_padding_mask)
        # l2l_z: B x Sl x d/2

        l2g_z = self.l2g_attention(Q, x_global, x_global, segment_ids, padding_mask=l2g_padding_mask)
        # l2g_z: B x Sl x d/2

        return torch.cat([l2l_z, l2g_z], dim=2)  # -> B x Sl x d

