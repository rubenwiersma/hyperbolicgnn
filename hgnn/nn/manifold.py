import abc
import torch
from torch.autograd import Function

from .poincare_distance import PoincareDistance
from ..util import dot, clamp_min, atanh


class Manifold(abc.ABC):
    r""" Abstract base class for a Manifold.
    """

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name
    
    @abc.abstractmethod
    def normalize(self, x):
        return NotImplementedError

    @abc.abstractmethod
    def dist(self, x, y):
        return NotImplementedError

    @abc.abstractmethod
    def log(self, y, x=None):
        return NotImplementedError

    @abc.abstractmethod
    def exp(self, v, x=None):
        return NotImplementedError

    @abc.abstractmethod
    def parallel_transport(self, v, x):
        return NotImplementedError


class EuclideanManifold(Manifold):

    def __init__(self, max_norm=1, EPS=1e-8):
        super().__init__('euclidean')
        self.max_norm = max_norm
        self.EPS = EPS

    def normalize(self, x):
        return torch.renorm(x, 2, 0, self.max_norm)

    def dist(self, x, y):
        return torch.sqrt(clamp_min(torch.sum((y - x).pow(2), dim=1), self.EPS))

    def log(self, y, x=None):
        return y if x is None else y - x

    def exp(self, v, x=None):
        return self.normalize(v) if x is None else self.normalize(v + x)

    def parallel_transport(self, v, x):
        return super().parallel_transport(v, x)


class PoincareBallManifold(Manifold):

    def __init__(self, EPS=1e-5):
        super().__init__('poincare')
        self.EPS = EPS

    def normalize(self, x):
        return torch.renorm(x, 2, 0, 1 - self.EPS)

    def mobius_add(self, x, y):
        y = y + self.EPS
        x_dot_y = dot(x, y)
        x_norm_squared = dot(x, x)
        y_norm_squared = dot(y, y)
        numerator = (1 + 2 * x_dot_y + y_norm_squared) * x + (1 - x_norm_squared) * y
        denominator = 1 + 2 * x_dot_y + x_norm_squared * y_norm_squared + self.EPS
        return self.normalize(numerator / denominator)

    def lambda_func(self, x):
        return 2 / (1 - dot(x, x))

    def dist(self, x, y):
        return PoincareDistance.apply(x, y, 1e-5)

    def log(self, y, x=None):
        if x is None:
            y = y + self.EPS
            norm_y = torch.linalg.norm(y, dim=-1, keepdim=True)
            return 1 / atanh(norm_y, self.EPS) * (y / norm_y)
        lambda_x = self.lambda_func(x)
        x_plus_y = self.mobius_add(-x, y) + self.EPS
        norm_x_plus_y = torch.linalg.norm(x_plus_y, dim=-1, keepdim=True)
        return 2 / lambda_x * atanh(norm_x_plus_y) * (x_plus_y / norm_x_plus_y)

    def exp(self, v, x=None):
        if x is None:
            v = v + self.EPS
            norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
            return self.normalize(torch.tanh(norm_v) * (v / norm_v))
        v = v + self.EPS
        lambda_x = self.lambda_func(x)
        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
        return self.mobius_add(x, torch.tanh(lambda_x * norm_v / 2) * (v / norm_v))

    def parallel_transport(self, v, x):
        return super().parallel_transport(v, x)


class LorentzManifold(Manifold):

    def __init__(self, EPS=1e-3, max_norm=1e-3, norm_clip=1):
        super().__init__('lorentz')
        self.EPS = EPS
        self.max_norm = max_norm
        self.norm_clip = 1

    def normalize(self, x):
        x = torch.renorm(x[:, 1:], 2, 0, self.max_norm)
        return torch.cat([torch.sqrt(1 + dot(x, x)), x], dim=-1)

    def normalize_tan(self, x, v):
        x = x[..., 1:]
        v = v[..., 1:]
        xv = torch.sum(x * v, dim=-1, keepdim=True)
        return torch.cat([xv / torch.sqrt(1 + dot(x, x)), v], dim=1)

    @staticmethod
    def scalar_product(x, y, keepdim=False):
        xy = x * y
        return torch.cat([-xy[..., 0:1], xy[..., 1:]], dim=-1).sum(dim=-1, keepdim=keepdim)

    def dist(self, x, y):
        d = -LorentzScalarProduct.apply(x, y)
        return Acosh.apply(d, self.EPS)

    def log(self, y, x=None):
        if x is None:
            x = torch.zeros_like(y)
            x[..., 0] = 1
        xy = self.scalar_product(x, y, keepdim=True)
        return Acosh.apply(-xy, self.EPS) / torch.sqrt(torch.clamp(xy.pow(2) - 1 + self.EPS, 1e-10)) * torch.addcmul(y, xy, x)

    def exp(self, v, x=None):
        if x is None:
            x = torch.zeros_like(v)
            x[..., 0] = 1
        v = self.normalize_tan(x, v)
        lorentz_norm_v = torch.sqrt(torch.clamp(self.scalar_product(v, v, keepdim=True) + self.EPS, 1e-10))
        lorentz_norm_v_clipped = torch.clamp(lorentz_norm_v, max=self.norm_clip)
        return self.normalize(torch.cosh(lorentz_norm_v_clipped) * x + torch.sinh(lorentz_norm_v_clipped) * (v / lorentz_norm_v))

    def nonlin_mapping(self, x, nonlin):
        # Map from Lorentz to Poincare ball
        x_poincare = x[..., 1:] / (x[..., 0:1] + 1)
        # Apply non-linearity
        x_nonlin = nonlin(x_poincare)
        # Map from Poincare ball to Lorentz
        squared_norm_x = dot(x_nonlin, x_nonlin)
        x_lorentz = torch.cat([1 + squared_norm_x, 2 * x_nonlin], dim=-1)
        return x_lorentz / (1 - squared_norm_x + self.EPS)

    def parallel_transport(self, v, x):
        return super().parallel_transport(v, x)

class LorentzScalarProduct(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return LorentzManifold.scalar_product(x, y)

    @staticmethod
    def backward(ctx, g):
        x, y = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(x).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * y, g * x

class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps):
        z = torch.sqrt(torch.clamp(x * x - 1 + eps, 1e-10))
        ctx.save_for_backward(z)
        ctx.eps = eps
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = torch.clamp(z, min=ctx.eps)
        z = g / z
        return z, None