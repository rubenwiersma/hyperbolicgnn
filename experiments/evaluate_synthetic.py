import sys
import os, os.path as osp
sys.path.append(os.getcwd())
import warnings
from progressbar import progressbar

import yaml
from easydict import EasyDict as edict
import argparse

import torch
from sklearn.metrics import f1_score

from torch_geometric.loader import DataLoader
from hgnn.transforms import OneHotDegree
from hgnn.datasets import SyntheticGraphs
from hgnn.models import GraphClassification
from hgnn.nn.manifold import EuclideanManifold, PoincareBallManifold, LorentzManifold

def test(args):
    dataset_root = osp.join(osp.dirname(osp.realpath(__file__)), 'data/SyntheticGraphs')
    pre_transform = OneHotDegree(args.in_features - 1, sum_in_out_degree=True, cat=False)
    test_dataset = SyntheticGraphs(dataset_root, split='test', pre_transform=pre_transform, node_num=(args.node_num_min, args.node_num_max), num_train=args.num_train, num_val=args.num_val, num_test=args.num_test)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.manifold == 'euclidean':
        manifold = EuclideanManifold()
    elif args.manifold == 'poincare':
        manifold = PoincareBallManifold()
    elif args.manifold == 'lorentz':
        manifold = LorentzManifold()
    else:
        manifold = EuclideanManifold()
        warnings.warn('No valid manifold was given as input, using Euclidean as default')
    model = GraphClassification(args, manifold).to(args.device)
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict)

    test_acc, test_f1 = evaluate(args, model, test_loader)
    print('Test accuracy {:.4f}, f1 score {:.4f}'.format(test_acc, test_f1))

def evaluate(args, model, data_loader):
    model.eval()
    correct = 0
    pred_list = []
    true_list = []
    for data in data_loader:
        data = data.to(args.device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        pred_list.append(pred.cpu().item())
        true_list.append(data.y.cpu().item())
    accuracy = correct / len(data_loader.dataset)
    f1 = f1_score(true_list, pred_list, average="macro")
    return accuracy, f1


if __name__ == "__main__":
    file_dir = osp.dirname(osp.realpath(__file__))

    # Parse arguments from command line
    parser = argparse.ArgumentParser('Synthetic Graph classification with Hyperbolic GNNs')
    parser.add_argument('--config', type=str, default=osp.join(file_dir, 'configs/synth_euclidean.yaml'), help='config file')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file to load from')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    terminal_args = parser.parse_args()

    # Parse arguments from config file
    with open(terminal_args.config) as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    # Additional arguments
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.checkpoint = terminal_args.checkpoint

    # Manual seed
    torch.manual_seed(terminal_args.seed)

    test(args)