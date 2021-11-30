import torch
from torch_geometric.utils import from_networkx, degree
from torch_geometric.data import InMemoryDataset

import numpy as np
import networkx as nx

class SyntheticGraphs(InMemoryDataset):

    def __init__(self, root, split='train',
                 node_num=(200, 500), num_train=2000, num_test=2000, num_val=2000,
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
            graph = from_networkx(nx.erdos_renyi_graph(num_node, np.random.uniform(0.01, 1)))
            graph.y = 0
            data_list.append(graph)
            if i % 100 == 0:
                print('{}/{}'.format(i, num_graphs))

        print("small_world")
        for i in range(num_graphs):
            num_node = np.random.randint(*self.node_num)
            graph = from_networkx(nx.watts_strogatz_graph(num_node, np.random.randint(low=1, high=200), np.random.uniform(0.01, 1)))
            graph.y = 1
            data_list.append(graph)
            if i % 100 == 0:
                print('{}/{}'.format(i, num_graphs))

        print("barabasi_albert")
        for i in range(num_graphs):
            num_node = np.random.randint(*self.node_num)
            graph = from_networkx(nx.barabasi_albert_graph(num_node, np.random.randint(low=1, high=200)))
            graph.y = 2
            data_list.append(graph)
            if i % 100 == 0:
                print('{}/{}'.format(i, num_graphs))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)
