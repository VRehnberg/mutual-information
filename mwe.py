import torch
from functorch import vmap

b, n, m = 3, 7, 2
x = torch.rand(b, n)
partition = torch.rand(n, m) > 0.5
vmap(lambda mask: x[:, mask].sum(-1))(partition)
