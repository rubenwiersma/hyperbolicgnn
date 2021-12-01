# Hyperbolic GNN replication

Replication of Hyperbolic Graph Neural Networks by Liu, Nickel and Kiela (https://arxiv.org/pdf/1910.12892.pdf) for Efficient Deep Learning winterschool 2021.

## Requirements
Version numbers show tested version. Some dependencies can be lower.
- PyTorch 1.10
- Pyg 2.0.2
- Requirements in `requirements.txt`

PyTorch and Pyg easiest to install with conda. Install the other requirements using the `requirements.txt` file.

## How to run
Run `$ python experiments/train_synthetic.py` from the root folder. To run all experiments for Table 1 in the paper, run `$ sh experiments/table_1.sh`.