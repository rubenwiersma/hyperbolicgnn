"""Base"""

from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
from torch import Tensor
from torch_geometric.nn import GCNConv

from .manifold import Manifold


class HGCNConv(GCNConv):
    r"""
    GCNConv which is modified to be agnostic of manifold geometries.
    """

    def __init__(self, in_channels: int, out_channels: int, manifold: Manifold,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(self, HGCNConv).__init__(in_channels, out_channels, improved, cached, add_self_loops, normalize, bias, **kwargs)

        self.manifold = manifold


def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        # TODO:
        # x = manifold.log(x)
        out = self.forward(x, edge_index, edge_weight)
        # out = manifold.exp(out)
        return out