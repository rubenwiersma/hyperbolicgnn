# %%
import os, sys
sys.path.append('../')

from hgnn.nn import EuclideanManifold, PoincareBallManifold, LorentzManifold

# %%
e = EuclideanManifold()
b = PoincareBallManifold()
l = LorentzManifold()

# %%
import torch
import matplotlib.pyplot as plt
xs = torch.meshgrid([torch.linspace(-10, 10, 50)] * 2, indexing="ij")
x = torch.stack([x.flatten() for x in xs], dim=1)
x = l.exp(x)
plt.figure(figsize=(10, 10))
plt.scatter(x[:, 0], x[:, 1])

# %%



