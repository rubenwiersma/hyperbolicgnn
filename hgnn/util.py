import torch

def dot(a, b):
    return torch.bmm(a.unsqueeze(-2), b.unsqueeze(-1))