import os.path as osp
import torch
from torch_geometric.utils import from_networkx, degree
from torch_geometric.data import InMemoryDataset

import numpy as np
import networkx as nx


class SyntheticGraphs(InMemoryDataset):
    r"""Synthetic graph dataset from the `"Hyperbolic Graph Neural networks"
    <https://arxiv.org/pdf/1910.12892.pdf>` paper, containing graphs generated with
    Erdos-Renyi, Watts-Strogatz, and Barabasi-Albert graph generation algorithms.
    Each graph is labelled with the algorithm that was used to generate the graph (0, 1, 2)
    and contains 100-500 nodes by default.

    Args:
        root (str): Root folder of the dataset.
        split (str, optional): Whether to use the train, val, or test split.
            (default: 'train') 
        node_num (tuple, optional): The range used to determine the number of nodes in each graph.
            (default: :obj:`(100, 200)`)
        num_train (int, optional): The number of graphs in the train set.
            (default: 2000)
        num_val (int, optional): The number of graphs in the validation set.
            (default: 2000)
        num_test (int, optional): The number of graphs in the test set.
            (default: 2000)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional):  A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed during processing.
            (default: :obj:`None`)
        pre_filter (callable, optional):  A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a filtered
            version. The data object will be filtered during processing.
            (default: :obj:`None`)
    """

    def __init__(self, root, split='train',
                 node_num=(100, 500), num_train=2000, num_val=2000, num_test=2000,
                 transform=None, pre_transform=None, pre_filter=None):
        self.node_num = node_num
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        super(SyntheticGraphs, self).__init__(root, transform, pre_transform, pre_filter)
        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        else:
            path = self.processed_paths[2]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        return

    def process(self):
        torch.save(self.generate_graphs(self.num_train), self.processed_paths[0])
        torch.save(self.generate_graphs(self.num_val), self.processed_paths[1])
        torch.save(self.generate_graphs(self.num_test), self.processed_paths[2])

    def generate_graphs(self, num_graphs):
        print("Generating graphs...")
        print("erdos_renyi")
        data_list = []
        for i in range(num_graphs):
            num_node = np.random.randint(*self.node_num)
            graph = from_networkx(nx.erdos_renyi_graph(num_node, np.random.uniform(0.1, 1)))
            graph.y = 0
            data_list.append(graph)
            if i % 500 == 0:
                print('{}/{}'.format(i, num_graphs))

        print("small_world")
        for i in range(num_graphs):
            num_node = np.random.randint(*self.node_num)
            graph = from_networkx(nx.watts_strogatz_graph(num_node, np.random.randint(low=2, high=100), np.random.uniform(0.1, 1)))
            graph.y = 1
            data_list.append(graph)
            if i % 500 == 0:
                print('{}/{}'.format(i, num_graphs))


        print("barabasi_albert")
        for i in range(num_graphs):
            num_node = np.random.randint(*self.node_num)
            graph = from_networkx(nx.barabasi_albert_graph(num_node, np.random.randint(low=2, high=100)))
            graph.y = 2
            data_list.append(graph)
            if i % 500 == 0:
                print('{}/{}'.format(i, num_graphs))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)
