# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://colab.research.google.com/github/VRehnberg/mutual-information/blob/main/mutual_information.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import sys
sys.path.append("torch-utils/src")


# %%
import torch
from torch import nn, linalg
from torch.autograd.functional import jacobian

import functorch

from torchutils import batched_jacobian
from torchutils.kmeans import kmeans
from torchutils.plotting import tensorshow
from torchutils.named_tensors import vmap

# %% [markdown]
# # Mutual Information
# This notebook was written to investigate a few different ways to estimate the mutual information between random variables from sampled data. This is then compared with the true mutual information.
# 
# 1. Analytical mutual information TODO
# 2. Jacobian/Hessian based mutual information TODO
# 3. Quantized/binned mutual information TODO
# %% [markdown]
# ## Network

# %%
class NormalLinear(nn.Linear):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        self.input_shape = input_shape
    
    def output_mutual_information(self, partition):
        '''
            partition (BoolTensor): n_modules × output_shape
        '''
        if (partition.int().sum(0) > 1).any():
            raise NotImplementedError("MI for overlapping modules not implemented.")

        weight, bias = self.parameters()
        #mean = bias
        cov_full = weight @ weight.T

        # Check ranks (this is unescessary if full rank)
        rank0 = linalg.matrix_rank(cov_full)
        rank1 = torch.sum(torch.hstack([
            linalg.matrix_rank(cov_full[mask, :][:,mask]) for mask in partition
        ]))
        if rank0 < rank1:
            return float("inf")
        elif rank1 != cov_full.size(0):
            raise NotImplementedError()
        
        # Compute MI
        det0 = cov_full.det()
        det1 = torch.prod(torch.hstack([
            cov_full[mask, :][:,mask].det() for mask in partition
        ]))
        return -0.5 * torch.log(det0 / det1)
        
    def sample(self, batch_shape):
        return torch.randn(batch_shape, self.input_shape, requires_grad=True)


# %%
n_modules = 2
in_size, out_size = (15, 7)
nl = NormalLinear(in_size, out_size)
partition = nn.functional.one_hot(torch.randint(n_modules, (out_size,))).bool().T
with torch.no_grad():
    print(nl.output_mutual_information(partition))

# %% [markdown]
# ## Mutual information

# %%
def jacobian_mutual_information(jac_full, jac_blocks):
    assert jac_full.ndim == 3

    # Covariances
    def det(jac):
        return jac.bmm(jac.transpose(1, 2)).det()

    det_full = det(jac_full)
    det_blocks = torch.hstack([det(jac_block).view(-1, 1) for jac_block in jac_blocks])

    # Local mutual information
    jmi = -0.5 * torch.log(det_full / torch.prod(det_blocks, 1))

    return jmi.mean(0)


# %%
jac_full = torch.rand(7, 10, 30)
partition = torch.randint(3, (10,))
jac_blocks = [jac_full[:, id==partition, :] for id in torch.unique(partition)]
jacobian_mutual_information(jac_full, jac_blocks)


# %%
def quantized_mutual_information(
    activations,
    partition,
    n_bins,
    cluster_method="kmeans",
    return_full=False,
):
    if partition.dtype != torch.bool:
        raise TypeError("Datatype of partition should be bool.")
    device = activations.device
    n_samples = activations.size("sample")
    n_neurons = activations.size("neuron")
    n_modules = partition.size("module")
    assert n_neurons == partition.size("neuron")
    assert n_samples >= n_bins

    if cluster_method=="kmeans":
        def quantize(points):
            # Cosine similarity
            points = (points - points.mean(0, keepdim=True)) / points.std(0, keepdim=True)
            return kmeans(points, n_bins)
    else:
        raise ValueError()

    quantized_activations = vmap(lambda mask: quantize(activations[:, mask]), ("module",))(
        partition
    )
    #print(partition.size(), partition.names)
    #print(activations.size(), activations.names)
    #functorch.vmap(lambda mask: quantize(activations[:, mask]), (0,), (1,))(
    #    partition
    #)
    #quantized_activations = torch.hstack([
    #    quantize(activations.masked_select(mask)) for mask in partition.unbind("neuron")
    #])
    
    # Compute pmfs
    activations_onehot = nn.functional.one_hot(quantized_activations).refine_names("bin").float()
    p_xy = activations_onehot @ activations_onehot.rename(neuron="neuron2", bin="bin2") / batch_size
    #torch.einsum("bij, bkl -> ikjl", activations_onehot, activations_onehot) / batch_size

    print(p_xy.names)
    #tensorshow(p_xy, )

    # Compute pairwise mutual information
    p_x = p_xy.diagonal(dim1="neuron", dim2="neuron2").diagonal(dim1="bin", dim2="bin2").refine_names("neuron", "bin")
    p_y = p_x.rename(bin="bin2", neuron="neuron2")
    #p_x = torch.einsum("iikk -> ik", p_xy)
    qmin = p_xy.div(p_x).div(p_y).pow(p_xy).log().sum(("neuron", "neuron2"))
    assert qmin.size("bin") == qmin.size("bin2") and qmin.ndim == 2
    return qmin
    

n_modules = 2
in_size, out_size = (15, 7)
activations = torch.rand(2000, out_size).refine_names("sample", "neuron")
partition = nn.functional.one_hot(
    torch.randint(n_modules, (out_size,))
).refine_names("neuron", "module").bool()
with torch.no_grad():
    print(quantized_mutual_information(activations, partition, 10))


# %%
n_modules = 2
in_size, out_size = (15, 2)
activations = torch.rand(2000, out_size)
partition = torch.eye(2, dtype=bool)
with torch.no_grad():
    print(quantized_mutual_information(activations, partition, 10))


# %%
# Gaussian test set-up
n_modules = 2
in_size, out_size = (15, 7)
batch_size = 2000
network = NormalLinear(in_size, out_size)
x = network.sample(batch_size)
activations = network(x)

partition = nn.functional.one_hot(torch.randint(n_modules, (out_size,))).bool().T


# %%

# True mutual information
mi = network.output_mutual_information(partition)
print(f"MI: {mi}")

# Local mutual information through Jacobian
jac_full = batched_jacobian(network, x)
jac_blocks = [jac_full[:, mask, :] for mask in partition]
lmi = jacobian_mutual_information(jac_full, jac_blocks)
print(f"LMI: {lmi}")

# Quantized mutual information through clustering
for k in range(2, 11, 2):
    with torch.no_grad():
        qmi = quantized_mutual_information(activations, partition, k)
    print(f"QMI k={k}: {qmi}")


