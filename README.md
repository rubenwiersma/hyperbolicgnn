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

The authors do not specify the architecture used to run this experiment and their published code uses two different architectures for the Euclidean manifold and the hyperbolic manifolds. We decided to use the same architecture for each manifold and base this architecture and the optimizer choices on the hyperbolic architectures in [their code](https://github.com/facebookresearch/hgnn/blob/master/params/SyntheticHyperbolicParams.py). We also use a different batch size from what the authors used: we use 32 graphs per batch, where the authors used 1 graph per batch. The larger batch size speeds up training and should give a better estimate of the loss per iteration.

F1 (macro) score on the synthetic graph dataset. The results with standard deviation from table 1 in the paper are listed between brackets.

| Manifold\Dim | 3                | 5                | 10               | 20               | 256              |
|--------------|------------------|------------------|------------------|------------------|------------------|
| Euclidean    | 95.1 (77.2±0.12) | 95.4 (90.0±0.21) | 96.4 (90.6±0.17) | 95.6 (94.8±0.25) | 95.8 (95.3±0.17) |
| Poincare     | 89.9 (93.0±0.05) | 94.9 (95.6±0.14) | 95.3 (95.9±0.14) | 96.1 (96.2±0.06) | 46.3 (93.7±0.05) |
| Lorentz      | 90.3 (94.1±0.03) | 95.6 (95.1±0.25) | 95.4 (96.4±0.23) | 96.4 (96.6±0.22) | 95.8 (95.3±0.28) |

### Discussion
For dimensions 5-20 in the hyperbolic embeddings, we find our results are comparable to the authors' published results. Our implementation has a higher F1 score in the Euclidean setting. So much so, that the Euclidean embedding is better than hyperbolic embeddings for most embedding sizes. An explanation for this difference could be that we use the same architecture for each embedding space, where the authors' implementation uses different architectures. It is unclear if these different architectures were also used for the results in the paper. To shed some light on this question, we re-trained the Euclidean setting with the architecture and configuration from the authors' code (see `experiments/configs/synth_euclidean_authors.yaml`) for the first four dimension sizes (256 dimensions is very costly, especially with the different architecture).

F1 (macro) score on the synthetic graph dataset with authors' architecture for Euclidean embedding.

| Manifold\Dim | 3                | 5                | 10               | 20               | 
|--------------|------------------|------------------|------------------|------------------|
| Euclidean    | 88.0 (77.2±0.12) | 87.8 (90.0±0.21) | 95.0 (90.6±0.17) | 90.4 (94.8±0.25) |

We also see a difference between our implementation on the hyperbolic settings with dimension 3 and 256. To study this difference, we retrained these settings with the exact implementation of the manifolds from the authors' code.

F1 (macro) score on the synthetic graph dataset with authors' manifold implementation.

| Manifold\Dim | 3                | 5                | 10               | 20               | 256              |
|--------------|------------------|------------------|------------------|------------------|------------------|
| Poincare     | - (93.0±0.05) | - (95.6±0.14) | - (95.9±0.14) | - (96.2±0.06) | - (93.7±0.05) |
| Lorentz      | - (94.1±0.03) | - (95.1±0.25) | - (96.4±0.23) | - (96.6±0.22) | - (95.3±0.28) |

## Credits
Network architecture and manifold mappings based on [Hyperbolic GNN implementation by the authors](https://github.com/facebookresearch/hgnn). Transforms and dataset built on [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric).
