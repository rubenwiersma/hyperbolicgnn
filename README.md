# Hyperbolic GNN replication

Replication of [Hyperbolic Graph Neural Networks by Liu, Nickel and Kiela](https://arxiv.org/pdf/1910.12892.pdf) for Efficient Deep Learning winterschool 2021.

## Installation
First install [PyTorch](https://pytorch.org) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) by following the installation instructions for each:

You can install the hgnn package either in your python environment:
```bash
pip install ./
```
or in the current folder with the `-e` flag specified.

## How to run
```bash
python experiments/train_synthetic.py
```
To run all experiments for Table 1 in the paper, run
```bash
sh experiments/table_1.sh
```

## Replication results
We set out to replicate the results in table 1 of [the paper](https://arxiv.org/pdf/1910.12892.pdf). The absolute results are expected to differ, because the dataset is generated at random - The exact dataset used for the results in the paper was not available. Still, we expect to see comparable relative results between the different manifold embeddings.

The authors do not specify the architecture used to run this experiment and their published code uses two different architectures for the Euclidean manifold and the hyperbolic manifolds. We decided to use the same architecture for each manifold and base this architecture and the optimizer choices on the hyperbolic architectures in [their code](https://github.com/facebookresearch/hgnn/blob/master/params/SyntheticHyperbolicParams.py). We also use a different batch size from what the authors used: we use 32 graphs per batch, where the authors used 1 graph per batch. The larger batch size speeds up training and should give a better estimate of the loss per iteration.

Table 1: F1 (macro) score on the synthetic graph dataset. The results with standard deviation from table 1 in the paper are listed between brackets.

| Manifold\Dim | 3                | 5                | 10               | 20               | 256              |
|--------------|------------------|------------------|------------------|------------------|------------------|
| Euclidean    | 95.1 (77.2±0.12) | 95.4 (90.0±0.21) | 96.4 (90.6±0.17) | 95.6 (94.8±0.25) | 95.8 (95.3±0.17) |
| Poincare     | 89.9 (93.0±0.05) | 94.9 (95.6±0.14) | 95.3 (95.9±0.14) | 96.1 (96.2±0.06) | 46.3 (93.7±0.05) |
| Lorentz      | 90.3 (94.1±0.03) | 95.6 (95.1±0.25) | 95.4 (96.4±0.23) | 96.4 (96.6±0.22) | 95.8 (95.3±0.28) |

### Discussion
For dimensions 5-20 in the hyperbolic embeddings, we find our results are comparable to the authors' published results. Our implementation has a higher F1 score in the Euclidean setting. So much so, that the Euclidean embedding is better than hyperbolic embeddings for most embedding sizes.

#### Architecture differences
An explanation for this difference could be that we use the same architecture for each embedding space, where the authors' implementation uses different architectures. It is unclear if these different architectures were also used for the results in the paper. To shed some light on this question, we re-trained the Euclidean setting with the architecture and configuration from the authors' code for the first four dimension sizes (see `experiments/configs/synth_euclidean_authors.yaml`).

Table 2: F1 (macro) score on the synthetic graph dataset with authors' architecture for Euclidean embedding.

| Manifold\Dim | 3                | 5                | 10               | 20               | 
|--------------|------------------|------------------|------------------|------------------|
| Euclidean    | 88.0 (77.2±0.12) | 87.8 (90.0±0.21) | 95.0 (90.6±0.17) | 90.4 (94.8±0.25) |

The different architecture used for the Euclidean setting results in lower scores. If we were to compare these results to the hyperbolic results in Table 1, we would draw the same conclusion as the paper: hyperbolic embeddings result in higher f1 scores on this task. It could be that this explains the difference between the paper and our replication results.

#### Manifold implementation
We also see a difference between our implementation on the hyperbolic settings with dimension 3 and 256. To find out if this difference is due to our implementation, we retrained these settings with the implementation of the logarithmic- and exponential-maps from the authors' code.

Table 3: F1 (macro) score on the synthetic graph dataset with authors' manifold implementation.

| Manifold\Dim | 3                | 5                | 10               | 20               | 256              |
|--------------|------------------|------------------|------------------|------------------|------------------|
| Poincare     | 91.3 (93.0±0.05) | 95.4 (95.6±0.14) | 95.9 (95.9±0.14) | 96.0 (96.2±0.06) | 65.6 (93.7±0.05) |
| Lorentz      | 94.3 (94.1±0.03) | 95.8 (95.1±0.25) | 96.3 (96.4±0.23) | 95.4 (96.6±0.22) | 94.8 (95.3±0.28) |

We observe that the logarithmic and exponential map implementations could explain some of the variance in the observed results for the lowest dimensionality, although the problem with high dimensions in the Poincare embedding is still not solved.

#### Dataset bias
It could be that some datasets are skewed to favor one embedding over the other. To study the effect of the random data generation, we performed 10 iterations of data generation, training, and evaluation. We used a smaller dataset for this experiment to lower the computational cost (200 instead of 2000 graphs) and only test for an embedding size of 3. The architectures and training settings are the same for each embedding.

Table 4: Average F1 (macro) score and standard deviation on the synthetic graph dataset over 10 random dataset generations and training rounds.

| Manifold\Dim | 3         |
|--------------|-----------|
| Euclidean    | 84.4±1.35 |
| Poincare     | 82.2±1.57 |
| Lorentz      | 85.1±2.45 |

### Conclusion

If we compare the results from Table 2 and Table 3, we could draw the same conclusion as the paper: hyperbolic geometries result in better accuracy on the synthetic dataset. However, we believe the results in Table 1 and Table 4 provide a fairer comparison, as we use the exact same architecture and optimizer for each setting. In that case, we do not see a significant difference between Euclidean and hyperbolic embeddings for this dataset.

## Credits
Network architecture, manifold mappings, and dataset generation based on [Hyperbolic GNN implementation by the authors](https://github.com/facebookresearch/hgnn).
