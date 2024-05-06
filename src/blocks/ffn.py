import torch.nn as nn
from torch import Tensor

from .. import utils


class SeqToBlocks(nn.Module):
    def __init__(self, block_len: int):
        self.block_len = block_len
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """TODO

        Args:
            x (Tensor): B' x S x d'

        Returns:
            Tensor: B' x NB x BL x d'
        """
        return utils.seq_to_blocks(x, self.block_len)


class ProjectionLayer(nn.Module):
    def __init__(
        self,
        fin: int,
        fout: int,
        nheads: int,
        bias: bool = False,
        identity: bool = False,
        block_len: int = 0,
    ):
        super().__init__()
        assert fout % nheads == 0

        self.nheads = nheads
        self.head_dim = fout // nheads
        self.nn = nn.Identity() if identity else nn.Linear(fin, fout, bias=bias)
        if block_len > 0:
            self.transform = SeqToBlocks(block_len)
        else:
            self.transform = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """_summary_

        Args:
            x (Tensor): B x S x d

        Returns:
            Tensor:
                (B * h) x S x (d / h)
                or
                (B * h) x NB x BL x (d / h)
        """
        batch_size, seq_len = x.shape[:2]
        out = (
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
            )  # -> (B * h) x S x (d / h)
        )

        return self.transform(out)
