# Hyperbolic GNN replication

Replication of [Hyperbolic Graph Neural Networks by Liu, Nickel and Kiela](https://arxiv.org/pdf/1910.12892.pdf) for Efficient Deep Learning winterschool 2021.

## Installation
First install [PyTorch](https://pytorch.org) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) by following the installation instructions for each:

You can install the hgnn package either in your python environment:
```
$ pip install ./
```
or in the current folder with the `-e` flag specified.

## How to run
```
$ python experiments/train_synthetic.py`
```
To run all experiments for Table 1 in the paper, run
```
$ sh experiments/table_1.sh
```

## Credits
Network architecture and manifold mappings based on [Hyperbolic GNN implementation by the authors](https://github.com/facebookresearch/hgnn). Transforms and dataset built on [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).