import torch.nn as nn
from torch import Tensor


class ProjectionLayer(nn.Module):
    def __init__(self, fin: int, fout: int, nheads: int, bias: bool = False, identity: bool = False):
        super().__init__()
        assert fout % nheads == 0

        self.nheads = nheads
        self.head_dim = fout // nheads
        self.nn = (
            nn.Identity()
            if identity
            else
            nn.Linear(fin, fout, bias=bias)
        )

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): B x S x d

        Returns:
            Tensor: (B * h) x S x (d / h)
        """
        batch_size, seq_len = x.shape[:2]
        return (
            self.nn(x)
            .view(
                batch_size,
                seq_len,
                self.head_dim,
                self.nheads,
            )
            .permute(0, 3, 1, 2)  # -> B x h x S x (d / h)
            .flatten(
                start_dim=0,
                end_dim=1,
            ) # -> (B * h) x S x (d / h)
        )
