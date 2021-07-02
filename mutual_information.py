#!/usr/bin/env python

import sys
sys.path.append("torch-utils/src")

import re

import torch
from torch import nn
from torch.autograd.functional import jacobian

from torchutils import batched_jacobian
from torchutils.kmeans import kmeans
from torchutils.named_tensors import lift_nameless, neinsum, ndiagonal
from torchutils.named_tensors import *
from torchutils.visualize import tensorshow

from network import NormalLinear


torch.Tensor.__oldrepr__ = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: re.sub("\s+", " " ,(
    "T(" +
        ",".join(f"{name}: {s}" for name, s in zip(self.names, self.size())) +
    ")" if self.nelement() > 9 else self.__oldrepr__()
))


def jacobian_mutual_information(jac_full, jac_blocks):
    assert jac_full.ndim == 3

    # Covariances
    def det(jac):
        return neinsum(jac, jac, sample=1, neuron=2).rename(None).det()

    det_full = det(jac_full)
    det_blocks = torch.hstack([det(jac_block).view(-1, 1) for jac_block in jac_blocks])

    # Local mutual information
    jmi = -0.5 * torch.log(det_full / torch.prod(det_blocks, 1))

    return jmi.mean(0)


#@profile
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
    assert n_samples >= n_bins
    assert n_neurons == partition.size("neuron")
    assert n_modules > 0

    if cluster_method=="kmeans":
        def quantize(points):
            # Cosine similarity
            points = (points - points.mean(0, keepdim=True)) / points.std(0, keepdim=True)
            return lift_nameless(kmeans, neuron=())(points, n_bins)
    else:
        raise ValueError()

    quantized_activations = lift_nameless(torch.stack)(
        [quantize(activations.index(mask, "neuron")) for mask in partition.unbind("module")],
        dim=-1,
    ).refine_names(..., "module")

    # Compute pmfs
    activations_onehot = lift_nameless(nn.functional.one_hot)(
        quantized_activations
    ).refine_names(..., "bin").float()
    for _ in range(100):
        p_xy = neinsum(activations_onehot, activations_onehot, module=2, bin=2) / n_samples
        #torch.einsum("bij, bkl -> ikjl", activations_onehot, activations_onehot) / batch_size

        tensorshow(p_xy, xdims=["module", "bin"], ydims=["module1", "bin1"])

        # Compute pairwise mutual information
        p_x = neinsum(activations_onehot, activations_onehot, module=1, bin=1) / n_samples
    p_y = p_x.rename(bin="bin1", module="module1").align_as(p_xy)
    p_x = p_x.align_as(p_xy)
    #p_x = torch.einsum("iikk -> ik", p_xy)
    qmin = p_xy.div(p_x).div(p_y).pow(p_xy).log().sum(("bin", "bin1"))  # TODO double check
    assert qmin.size("module") == qmin.size("module1") and qmin.ndim == 2
    return qmin


def gaussian_test(n_modules=2, in_size=15, out_size=7, batch_size=2000):
    # Gaussian test set-up
    network = NormalLinear(in_size, out_size)
    x = network.sample(batch_size)
    activations = network(x).rename("sample", "neuron")

    # Partition the network
    partition = nn.functional.one_hot(torch.randint(n_modules, (out_size,))).bool().T
    partition.rename_("module", "neuron")

    # True mutual information
    mi = lift_nameless(network.output_mutual_information, out_names=())(partition)
    print(f"MI: {mi}")

    # Local mutual information through Jacobian
    jac_full = batched_jacobian(network, x).rename("sample", "neuron", "input")
    jac_blocks = [index(jac_full, mask, "neuron") for mask in partition]
    lmi = jacobian_mutual_information(jac_full, jac_blocks)
    print(f"LMI: {lmi}")

    # Quantized mutual information through clustering
    for k in [10]:
        with torch.no_grad():
            qmi = quantized_mutual_information(activations, partition, k)
        print(f"QMI k={k}: {qmi}")


def main():
    
    # Try QMI
    #n_modules = 2
    #in_size, out_size = (150, 7)
    #activations = torch.rand(2000, out_size).refine_names("sample", "neuron")
    #partition = nn.functional.one_hot(
    #    torch.randint(n_modules, (out_size,))
    #).refine_names("neuron", "module").bool()
    #with torch.no_grad():
    #    print(quantized_mutual_information(activations, partition, 10))
    
    gaussian_test()

    from matplotlib import pyplot as plt


if __name__ == "__main__":
    main()
