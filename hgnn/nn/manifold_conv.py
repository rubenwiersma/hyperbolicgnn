from typing import Callable
import torch
from torch import Tensor

from .manifold import Manifold, EuclideanManifold

class ManifoldConv(torch.nn.Module):
    r"""
    Convolution wrapper for Hyperbolic Graph Neural Networks.
    Instantiate this convolution with a Manifold object (e.g.: Euclidean, Poincare Ball, Lorentz) and a convolution module.
    The forward pass will apply the mapping to- and from the Euclidean tangent plane, where the convolution is applied.
    If the manifold requires a mapping to another manifold for non-linearities,
    it will use this mapping to apply the non-linearity.
    """

    def __init__(self, conv: Callable, manifold: Manifold = EuclideanManifold, nonlin=torch.nn.ReLU):
        super(self, ManifoldConv).__init__()
        self.conv = conv
        self.nonlin = nonlin
        self.manifold = manifold

    def forward(self, x, *args) -> Tensor:
        x = self.manifold.log(x)
        out = self.conv(x, *args)
        out = self.manifold.exp(out)
        if hasattr(self.manifold, 'nonlin_mapping'):
            out = self.manifold.nonlin_mapping(out, self.nonlin)
        else:
            out = self.nonlin(out)
        return out