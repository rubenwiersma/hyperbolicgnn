from typing import Callable
import torch
from torch import Tensor

from .manifold import Manifold

class ManifoldConv(torch.nn.Module):
    r"""
    Convolution wrapper which applies the necessary transport maps.
    """

    def __init__(self, conv: Callable, manifold: Manifold, nonlin=torch.nn.ReLU):
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