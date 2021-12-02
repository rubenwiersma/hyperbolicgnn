from typing import Callable
import torch
from torch import Tensor

from .manifold import Manifold, EuclideanManifold


class ManifoldConv(torch.nn.Module):
    r"""Convolution wrapper for Hyperbolic Graph Neural Networks.
    Instantiate this convolution with a Manifold object (e.g.: Euclidean, Poincare Ball, Lorentz) and a convolution module.
    The forward pass will apply the mapping to- and from the Euclidean tangent plane, where the convolution is applied.
    If the manifold requires a mapping to another manifold for non-linearities,
    it will use this mapping to apply the non-linearity.

    Args:
        conv (callable): Convolution to apply.
        manifold (Manifold): The manifold used as embedding space.
            (default: :obj:`EuclideanManifold`)
        nonlin (callable, optional): Nonlinearity that is applied in the embedding space.
            (default: :obj:`SELU`)
        dropout (float, optional): Dropout applied after the convolution and before exponential map.
            (default: 0)
        from_euclidean (bool, optional): If set to :obj:`True`, does not apply logarithmic map
            before applying the convolution.
            (default: :obj:`False`)
    """

    def __init__(self, conv: Callable, manifold: Manifold=EuclideanManifold(), nonlin=torch.nn.SELU(), dropout=0, from_euclidean=False):
        super(ManifoldConv, self).__init__()
        self.conv = conv
        self.nonlin = nonlin
        self.manifold = manifold
        self.dropout = torch.nn.Dropout(dropout)
        self.from_euclidean = from_euclidean

    def forward(self, x, *args) -> Tensor:
        x = self.manifold.log(x) if not self.from_euclidean else x
        out = self.conv(x, *args)
        if self.manifold.name == 'lorentz':
            out[:, 0] = 0
        out = self.dropout(out)
        out = self.manifold.exp(out)
        if hasattr(self.manifold, 'nonlin_mapping'):
            out = self.manifold.nonlin_mapping(out, self.nonlin)
        else:
            out = self.nonlin(out)
        return out