import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from ..nn.centroid_distance import CentroidDistance
from ..nn import Manifold, ManifoldConv, EuclideanManifold


class GraphClassification(nn.Module):
    r"""
    Graph classification network based on Hyperbolic GNN paper.
    """

    def __init__(self, args, manifold: Manifold = EuclideanManifold()):
        super(GraphClassification, self).__init__()
        self.manifold = manifold

        self.embedding = nn.Linear(args.in_features, args.embed_dim, bias=False)
        nn.init.xavier_uniform_(self.embedding.weight.data)

        self.layers = torch.nn.ModuleList()
        for _ in range(args.num_layers):
            conv = GCNConv(args.embed_dim, args.embed_dim)
            self.layers.append(ManifoldConv(conv, manifold, dropout=args.dropout))

        self.centroid_distance = CentroidDistance(args.num_centroid, args.embed_dim, manifold)

        self.output_linear = nn.Linear(args.num_centroid, args.num_class)
        nn.init.xavier_uniform_(self.output_linear.weight.data)


    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, edge_index)
        centroid_dist = self.centroid_distance(x, batch=data.batch)
        return self.output_linear(centroid_dist)