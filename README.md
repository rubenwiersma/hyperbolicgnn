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

## Replication results
We set out to replicate the results in table 1 of [the paper](https://arxiv.org/pdf/1910.12892.pdf). The absolute results are expected to differ, because the dataset is generated at random - The exact dataset used for the results in the paper was not available. Still, we expect to see comparable relative results between the different manifold embeddings.

The authors do not specify the architecture used to run this experiment and their published code uses two different architectures for the Euclidean manifold and the hyperbolic manifolds. We decided to use the same architecture for each manifold and base this architecture and the optimizer choices on the hyperbolic architectures in [their code](https://github.com/facebookresearch/hgnn/blob/master/params/SyntheticHyperbolicParams.py).

### Our results

| Manifold\Dim | 3 | 5 | 10 | 20 | 256 |
|--------------|---|---|----|----|-----|
| Euclidean    |   |   |    |    |     |
| Poincare     |   |   |    |    |     |
| Lorentz      |   |   |    |    |     |

### Paper results
F1 (macro) score and standard deviation on the synthetic graph dataset.

| Manifold\Dim | 3         | 5         | 10        | 20        | 256       |
|--------------|-----------|-----------|-----------|-----------|-----------|
| Euclidean    | 77.2±0.12 | 90.0±0.21 | 90.6±0.17 | 94.8±0.25 | 95.3±0.17 |
| Poincare     | 93.0±0.05 | 95.6±0.14 | 95.9±0.14 | 96.2±0.06 | 93.7±0.05 |
| Lorentz      | 94.1±0.03 | 95.1±0.25 | 96.4±0.23 | 96.6±0.22 | 95.3±0.28 |

## Credits
Network architecture and manifold mappings based on [Hyperbolic GNN implementation by the authors](https://github.com/facebookresearch/hgnn). Transforms and dataset built on [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).
