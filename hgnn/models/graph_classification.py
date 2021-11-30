import torch
import torch.nn as nn
from ..nn import Manifold, EuclideanManifold

class GraphClassification(nn.Module):

    def __init__(self, in_channels, embed_size, manifold=EuclideanManifold()):
        self.in_channels = in_channels
        self.embed_size = embed_size
        self.manifold = manifold