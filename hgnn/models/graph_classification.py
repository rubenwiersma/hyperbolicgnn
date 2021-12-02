import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from ..nn.centroid_distance import CentroidDistance
from ..nn import Manifold, ManifoldConv, EuclideanManifold


class GraphClassification(nn.Module):
    r"""
    Graph classification network based on Hyperbolic GNN paper.

    Args:
        args (dict): Arguments supplied to network.
        manifold (Manifold): Manifold to use as embedding space.
            (default: :obj:`EuclideanManifold`)
    """

    def __init__(self, args, manifold: Manifold = EuclideanManifold()):
        super(GraphClassification, self).__init__()
        self.manifold = manifold

        self.embedding = nn.Linear(args.in_features, args.embed_dim, bias=False)
        if args.weight_init and args.weight_init == 'xavier':
            nn.init.xavier_uniform_(self.embedding.weight.data)

        self.layers = torch.nn.ModuleList()
        for i in range(args.num_layers):
            conv = GCNConv(args.embed_dim, args.embed_dim, bias=False)
            if args.weight_init and args.weight_init == 'xavier':
                nn.init.xavier_uniform_(conv.lin.weight.data)
            self.layers.append(ManifoldConv(conv, manifold, dropout=args.dropout, from_euclidean=i == 0))

        self.centroid_distance = CentroidDistance(args.num_centroid, args.embed_dim, manifold, args.weight_init)

        self.output_linear = nn.Linear(args.num_centroid, args.num_class)
        if args.weight_init and args.weight_init == 'xavier':
            nn.init.xavier_uniform_(self.output_linear.weight.data)
            nn.init.uniform_(self.output_linear.bias.data, -1e-4, 1e-4)


    def forward(self, data):
        x = self.embedding(data.x)
        for layer in self.layers:
            x = layer(x, data.edge_index)
        centroid_dist = self.centroid_distance(x, batch=data.batch)
        return self.output_linear(centroid_dist)