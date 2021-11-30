import abc
import torch
from ..util import dot


class Manifold(abc.ABC):
    r""" Abstract base class for a Manifold.
    """

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def init_embed(self, embed, range=1e-3):
        embed.weight.data.uniform_(-range, range)
        embed.weight.data.copy_(self.normalize(embed.weight.data))
    
    @abc.abstractmethod
    def normalize(self, x):
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

    @abc.abstractmethod
    def dist(self, x, y):
        return NotImplementedError


class EuclideanManifold(Manifold):

    def __init__(self, max_norm=1):
        super().__init__('Euclidean')
        self.max_norm = max_norm

    def normalize(self, x):
        return torch.renorm(x, 2, 0, self.max_norm)

    def log(self, y, x=None):
        return y if x is None else y - x

    def exp(self, v, x=None):
        return v if x is None else v + x

    def parallel_transport(self, v, x):
        return v - x

    def dist(self, x, y):
        return torch.linalg.norm(x - y, dim=-1)


class PoincareBallManifold(Manifold):

    def __init__(self, EPS=1e-5):
        super().__init__('Poincare Ball')
        self.EPS = EPS

    def init_embed(self, embed, range=1e-2):
        return super().init_embed(embed, range=range)

    def normalize(self, x):
        return torch.renorm(x, 2, 0, 1 - self.EPS)

    def mobius_add(self, x, y):
        x_dot_y = dot(x, y)
        x_norm_squared = dot(x, x)
        y_norm_squared = dot(y, y)
        numerator = (1 + 2 * x_dot_y + y_norm_squared) * x + (1 - x_norm_squared) * y
        denominator = 1 + 2 * x_dot_y + x_norm_squared * y_norm_squared
        return self.normalize(numerator / denominator)

    def lambda_func(self, x):
        return 2 / (1 - dot(x, x))

    def log(self, y, x=None):
        if x is None:
            norm_y = torch.linalg.norm(y, dim=-1, keepdim=True)
            return torch.arctanh(norm_y) * (y / norm_y.clip(self.EPS))
        lambda_x = self.lambda_func(x)
        x_plus_y = - self.mobius_add(x, y)
        norm_x_plus_y = torch.linalg.norm(x_plus_y, dim=-1, keepdim=True)
        return 2 / lambda_x * torch.arctanh(norm_x_plus_y) * (x_plus_y / norm_x_plus_y.clip(self.EPS))

    def exp(self, v, x=None):
        if x is None:
            norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
            return self.normalize(torch.tanh(norm_v) * (v / norm_v.clip(self.EPS)))
        lambda_x = self.lambda_func(x)
        norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
        return self.mobius_add(x, torch.tanh(lambda_x * norm_v / 2) * (v / norm_v.clip(self.EPS)))

    def parallel_transport(self, v, x):
        return NotImplementedError

    def dist(self, x, y):
        x_y = x - y
        return torch.arccosh(1 + 2 * (dot(x_y, x_y) / ((1 - dot(x, x)) * (1 - dot(y, y)))))


class LorentzManifold(Manifold):

    def __init__(self, EPS=1e-3, max_norm=1e-3):
        super().__init__('Lorentz')
        self.EPS = EPS
        self.max_norm = max_norm

    def init_embed(self, embed, range=1e-2):
        return super().init_embed(embed, range=range)

    def normalize(self, x):
        d = x.size(-1) - 1
        x_narrowed = x.narrow(-1, 1, d)
        x_narrowed = torch.renorm(x_narrowed, 2, 0, self.max_norm)
        return torch.cat([torch.sqrt(1 + dot(x_narrowed, x_narrowed)), x_narrowed], dim=-1) 

    def scalar_product(self, x, y, keepdim=True):
        return -(x[..., 0:1] * y[..., 0:1]) + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)

    def log(self, y, x=None):
        if x is None:
            x = torch.zeros_like(y)
            x[..., 0] = 1
        return (self.dist(x, y) / torch.sqrt(self.scalar_product(x, y).pow(2) - 1).clip(self.EPS)) * (y + self.scalar_product(x, y) * x)

    def exp(self, v, x=None):
        if x is None:
            x = torch.zeros_like(v)
        lorentz_norm_v = torch.sqrt(self.scalar_product(v, v))
        return self.normalize(torch.cosh(lorentz_norm_v) * x + torch.sinh(lorentz_norm_v) * (v / lorentz_norm_v.clip(self.EPS)))

    def parallel_transport(self, v, x):
        return NotImplementedError

    def dist(self, x, y):
        return torch.arccosh(-self.scalar_product(x, y))

    def nonlin_mapping(self, x, nonlin):
        # Map from Lorentz to Poincare ball
        x_poincare = x[..., 1:] / (x[..., 0:1] + 1)
        # Apply non-linearity
        x_nonlin = nonlin(x_poincare)
        # Map from Poincare ball to Lorentz
        squared_norm_x = dot(x_nonlin, x_nonlin)
        x_lorentz = torch.cat([1 + squared_norm_x, 2 * x_nonlin[..., 1:]], dim=-1)
        return x_lorentz / (1 - squared_norm_x).clip(self.EPS)