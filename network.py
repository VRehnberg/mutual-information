import torch
from torch import nn, linalg


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


if __name__=="__main__":
    n_modules = 2
    in_size, out_size = (15, 7)
    nl = NormalLinear(in_size, out_size)
    partition = nn.functional.one_hot(torch.randint(n_modules, (out_size,))).bool().T
    with torch.no_grad():
        print(nl.output_mutual_information(partition))
