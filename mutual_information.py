import sys
sys.path.append("torch-utils/src")

import torch
from torch import nn
from torch.autograd.functional import jacobian

from torchutils import batched_jacobian
from torchutils.kmeans import kmeans
from torchutils.named_tensors import lift_nameless
from torchutils.named_tensors import *

from network import NormalLinear


torch.Tensor.__repr__ = lambda self: ", ".join(f"{name}: {s}" for name, s in zip(self.names, self.size()))


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
    p_xy = neinsum(activations_onehot, activations_onehot, sample=0, module=2, bin=2) / n_samples
    #torch.einsum("bij, bkl -> ikjl", activations_onehot, activations_onehot) / batch_size

    print(p_xy.names)
    #tensorshow(p_xy, )

    # Compute pairwise mutual information
    p_x = p_xy.diagonal(dim1="module", dim2="module2").diagonal(dim1="bin", dim2="bin2").refine_names("neuron", "bin")
    p_y = p_x.rename(bin="bin2", module="module2")
    #p_x = torch.einsum("iikk -> ik", p_xy)
    qmin = p_xy.div(p_x).div(p_y).pow(p_xy).log().sum(("neuron", "neuron2"))
    assert qmin.size("bin") == qmin.size("bin2") and qmin.ndim == 2
    return qmin


def gaussian_test(n_modules=2, in_size=15, out_size=7, batch_size=2000):
    # Gaussian test set-up
    network = NormalLinear(in_size, out_size)
    x = network.sample(batch_size)
    activations = network(x)

    # Partition the network
    partition = nn.functional.one_hot(torch.randint(n_modules, (out_size,))).bool().T

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


def main():
    
    # Try QMI
    n_modules = 2
    in_size, out_size = (150, 7)
    activations = torch.rand(2000, out_size).refine_names("sample", "neuron")
    partition = nn.functional.one_hot(
        torch.randint(n_modules, (out_size,))
    ).refine_names("neuron", "module").bool()
    with torch.no_grad():
        print(quantized_mutual_information(activations, partition, 10))
    
    #gaussian_test()


if __name__ == "__main__":
    main()