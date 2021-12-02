import torch

def dot(a, b):
    return torch.bmm(a.unsqueeze(-2), b.unsqueeze(-1)).squeeze(-1)

def atanh(x, EPS):
	values = torch.min(x, torch.Tensor([1.0 - EPS]).to(x.device))
	return 0.5 * (torch.log(1 + values + EPS) - torch.log(1 - values + EPS))

def clamp_min(x, min_value):
	t = torch.clamp(min_value - x.detach(), min=0)
	return x + t