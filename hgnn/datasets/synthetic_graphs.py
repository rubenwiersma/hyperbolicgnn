from torch_geometric.data import InMemoryDataset

class SyntheticGraphs(InMemoryDataset):

    def __init__(self, path):
        super().__init__()
        self.path = path