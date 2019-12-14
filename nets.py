import torch
import math
import random
import collections
import itertools
import json
import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):
    def __init__(self, input_dim=(32*32*3), r_0=None, d=None):
        super(FCNet, self).__init__()
        self.input_dim = input_dim
        self.param_sizes = \
            [self.input_dim*200, 200, 200*200, 200, 200*10, 10]
        self.r_xavier = math.sqrt(200 + 200 + 10)
        self.D = sum(self.param_sizes)

        if r_0 is None:
            r_0 = self.r_xavier
        w = torch.randn(self.D)
        w *= r_0 / w.norm()
        self.weights = nn.Parameter(w)

        # self.w1 = nn.Parameter(torch.randn(input_dim, 200) / math.sqrt(input_dim))
        # self.b1 = nn.Parameter(torch.zeros(200))
        # self.w2 = nn.Parameter(torch.randn(200, 200) / math.sqrt(200))
        # self.b2 = nn.Parameter(torch.zeros(200))
        # self.w3 = nn.Parameter(torch.randn(200, 10) / math.sqrt(200))
        # self.b3 = nn.Parameter(torch.zeros(10))

    def get_params(self):
        return self.weights

    def get_radius(self):
        with torch.no_grad():
            params = self.get_params()
            return params.norm().item()

    def get_radii(self):
        with torch.no_grad():
           params = self.get_params()
           return [param.norm().item() for param in 
                   params.split(self.param_sizes)]

    def forward(self, x):
        params = self.get_params()
        w1, b1, w2, b2, w3, b3 = params.split(self.param_sizes)

        w1 = w1.view(self.input_dim, 200)
        w2 = w2.view(200, 200)
        w3 = w3.view(200, 10)

        x = x.view(-1, self.input_dim)
        x = F.relu(x @ w1 + b1)
        x = F.relu(x @ w2 + b2)
        x = x @ w3 + b3

        return x


class RandomHyperplaneNet(FCNet):
    def __init__(self, d=None, r_0=None, input_dim=(32*32*3)):
        nn.Module.__init__(self)
        self.input_dim = input_dim
        self.d = d
        self.param_sizes = \
            [self.input_dim*200, 200, 200*200, 200, 200*10, 10]
        self.r_xavier = math.sqrt(200 + 200 + 10)
        self.D = sum(self.param_sizes)

        if d is None:
            d = 100
        if r_0 is None:
            r_0 = self.r_xavier

        self.make_params(r_0)

    def make_params(self, r_0):
        # The coordinates in the d-dimensional subspace
        self.weights = nn.Parameter(torch.zeros(self.d))
        # The fixed offset P into parameter space
        self.offset = torch.randn(self.D, device='cuda')
        self.offset *= r_0 / self.offset.norm()
        # Transformation matrix from d subspace into full D space
        self.M = self.get_random_ortho_matrix(self.D, self.d)

    def get_random_ortho_matrix(self, D, d):
        # Use the procedure in the paper for a sparse random projection
        M = torch.zeros(D, d, device='cuda')
        for i in range(d):
            col = torch.zeros(D)
            prob = 1 / math.sqrt(D)
            col[torch.rand(D) < prob] = 1
            col[torch.rand(D) < 0.5] *= -1
            col /= col.norm()
            M[:,i] = col
        return M

    def get_params(self):
        return self.offset + self.M @ self.weights


class SparseRandomHyperplaneNet(RandomHyperplaneNet):
    def __init__(self, **kwargs):
        super(SparseRandomHyperplaneNet, self).__init__(**kwargs)

    def get_random_ortho_matrix(self, D, d):
        prob = 1 / math.sqrt(D)
        all_idxs = torch.tensor([], dtype=int, device='cuda')
        all_vals = torch.tensor([], device='cuda')
        # Build up the matrix column by column
        for i in range(d):
            # First generate the nonzero indices in this column.
            # Profiling told me that this was the slowest step. So I
            # moved the random number generation from the CPU to GPU.
            rands = torch.cuda.FloatTensor(D).random_(to=int(1/prob))
            row_idxs = (rands == 0).nonzero().view(-1)
            n = len(row_idxs)

            col_idxs = torch.ones(n, dtype=int, device='cuda') * i
            idxs = torch.stack((row_idxs, col_idxs))
            all_idxs = torch.cat((all_idxs, idxs), dim=1)

            vals = torch.ones(n, device='cuda') / math.sqrt(n)
            vals[torch.rand(n) < 0.5] *= -1
            all_vals = torch.cat((all_vals, vals))

        return torch.sparse_coo_tensor(
                indices=all_idxs, values=all_vals, size=(D,d),
                device='cuda')

    def get_params(self):
        # For some reason the sparse implementation can only do
        # matrix-matrix multiplication, not matrix-vector multiplication
        # So I need to do some reshaping business.
        w = self.weights.view(-1, 1)
        return self.offset + torch.sparse.mm(self.M, w).view(-1)


class RadiusConstrain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # Need to do dummy operation so that it's not optimized out
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # Subtract off the radial component of the gradient
        r_hat = input / input.norm()
        grad_input = grad_output.clone()
        grad_input -= torch.dot(r_hat, grad_output) * r_hat
        return grad_input

radius_constrain = RadiusConstrain.apply


class HypersphereFCNet(FCNet):

    def get_params(self):
        return radius_constrain(self.weights)


class SparseHypersphereNet(SparseRandomHyperplaneNet):

    def make_params(self, r_0):
        # The coordinates in the d-dimensional subspace
        w = torch.randn(self.d, device='cuda')
        w *= r_0 / w.norm()
        self.weights = nn.Parameter(w)
        # Transformation matrix from d subspace into full D space
        self.M = self.get_random_ortho_matrix(self.D, self.d)

    def get_params(self):
        w = radius_constrain(self.weights).view(-1, 1)
        return torch.sparse.mm(self.M, w).view(-1)
        # params = torch.sparse.mm(self.M, self.weights).view(-1)
        # return radius_constrain(params)


# Too long of a function name
def f(D, d):
    return SparseRandomHyperplaneNet.get_random_ortho_matrix(None, D, d)
