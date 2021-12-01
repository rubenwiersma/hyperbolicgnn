import sys
import os, os.path as osp
sys.path.append(os.getcwd())
import time
import warnings
from progressbar import progressbar

import yaml
from easydict import EasyDict as edict
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree
from hgnn.datasets import SyntheticGraphs
from hgnn.models import GraphClassification
from hgnn.nn.manifold import EuclideanManifold, PoincareBallManifold, LorentzManifold

def train(args):
    dataset_root = osp.join(osp.dirname(osp.realpath(__file__)), 'data/SyntheticGraphs')
    pre_transform = OneHotDegree(args.in_features - 1, cat=False)
    train_dataset = SyntheticGraphs(dataset_root, split='train', pre_transform=pre_transform, node_num=(args.node_num_min, args.node_num_max), num_train=args.num_train, num_val=args.num_val,  num_test=args.num_test)
    val_dataset = SyntheticGraphs(dataset_root, split='val', pre_transform=pre_transform, node_num=(args.node_num_min, args.node_num_max), num_train=args.num_train, num_val=args.num_val, num_test=args.num_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.manifold == 'Euclidean':
        manifold = EuclideanManifold()
    elif args.manifold == 'Poincare':
        manifold = PoincareBallManifold()
    elif args.manifold == 'Lorentz':
        manifold = LorentzManifold()
    else:
        manifold = EuclideanManifold
        warnings.warn('No valid manifold was given as input, using Euclidean as default')
    model = GraphClassification(args, manifold).to(args.device)

    # TODO add optimizer for hyperbolic parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=args.optimizer == 'amsgrad', weight_decay=args.weight_decay)
    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

    best_accuracy = 0
    for epoch in range(args.epochs):
        model.train()

        total_loss = 0
        for data in progressbar(train_loader):
            data = data.to(args.device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_function(out, data.y)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        val_acc = evaluate(args, model, val_loader)
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), osp.join(args.logdir, 'best.pt'))
            best_accuracy = val_acc
        args.writer.add_scalar('training loss', total_loss, epoch)
        args.writer.add_scalar('validation accuracy', val_acc, epoch)


def evaluate(args, model, data_loader):
    model.eval()
    correct = 0
    for data in data_loader:
        data = data.to(args.device)
        with torch.no_grad():
            pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(data_loader.dataset)


if __name__ == "__main__":
    file_dir = osp.dirname(osp.realpath(__file__))

    # Parse arguments from command line
    parser = argparse.ArgumentParser('Synthetic Graph classification with Hyperbolic GNNs')
    parser.add_argument('--config', type=str, default=osp.join(file_dir, 'configs/synth.yaml'), help='config file')
    terminal_args = parser.parse_args()

    # Parse arguments from config file
    with open(terminal_args.config) as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    # Additional arguments
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create log directory
    experiment_name = 'hgnn_{}_dim{}'.format(args.manifold, args.embed_dim)
    run_time = time.strftime("%d%b%y_%H_%M", time.localtime(time.time()))
    args.logdir = osp.join(file_dir, 'logs', experiment_name, run_time)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    # Setup tensorboard writer
    args.writer = SummaryWriter(args.logdir)

    # Manual seed
    torch.manual_seed(42)

    train(args)