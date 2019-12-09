import torch
import math
import random
import collections
import itertools
import torch.nn as nn
import torch.nn.functional as F

import cifar10
import mnist

train_dl = cifar10.get_trainloader()
test_dl = cifar10.get_testloader()

# train_dl = mnist.get_trainloader()
# test_dl = mnist.get_testloader()

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


# Too long of a function name
def f(D, d):
    return SparseRandomHyperplaneNet.get_random_ortho_matrix(None, D, d)

def train(n_epochs=10, epoch_start=0,
         Net=FCNet, net=None, log=None, **kwargs):

    if net is None:
        print('Making Network...')
        net = Net(**kwargs)
        net.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    if log is None:
        log = collections.defaultdict(list)
    print_every = 100
    examples_seen = 0
    batch_size = train_dl.batch_size
    n_examples = len(train_dl.dataset.targets)

    def get_test_acc():
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dl:
                images, labels = data
                images = images.to('cuda')
                labels = labels.to('cuda')
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct/total 

    log['test_acc'].append((epoch_start, get_test_acc()))
    log['radius'].append((epoch_start, net.get_radius()))
    # TODO get the training loss at initialization as well!

    print('Starting Training...')
    for epoch in range(epoch_start, epoch_start + n_epochs):
        running_loss = 0
        for i, data in enumerate(train_dl):
            inputs = data[0].to('cuda')
            labels = data[1].to('cuda')

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            examples_seen += len(labels)
            running_loss += loss.item()

            # print statistics
            if i % print_every == (print_every - 1):
                avg_loss = running_loss / print_every
                t = epoch + (i * batch_size / n_examples)
                running_loss = 0

                r = net.get_radius()
                log['loss'].append((t, avg_loss))
                log['radius'].append((t, r))

                if epoch < 1:
                    test_acc = get_test_acc()
                    log['test_acc'].append((t, test_acc))
                    print('[Epoch %.2f] loss: %.3f, radius: %.3f, test accuracy: %.2f%%' % (t, avg_loss, r, test_acc * 100))
                else:
                    print('[Epoch %.2f] loss: %.3f, radius: %.3f' % (t, avg_loss, r), end='\r')

        t = epoch + 1
        test_acc = get_test_acc()
        log['test_acc'].append((t, test_acc))
        # avg_loss = running_loss / (i % print_every)
        # log['loss'].append((t, avg_loss))
        r = net.get_radius()
        log['radius'].append((t, r))
        rs = net.get_radii()
        log['radii'].append([t] + rs)

        print('[Epoch %d] loss: %.3f, radius: %.3f, test accuracy: %.2f%%' % (t, avg_loss, r, test_acc * 100))

    print('Finished Training')
    return log,net

logs = []
for d in [300, 1000, 3000, 7000, 10000]:
    for r_0 in [0.2, 20, 50, 200]:
        print('--- d = %d, r_0 = %d ---' % (d, r_0))
        log, net = train(n_epochs=20, r_0=r_0, d=d,
                        Net=SparseRandomHyperplaneNet)
        log['d'] = d
        log['r_0'] = r_0
        logs.append(log)

# logs = []
# for r_0 in [0.2, 2, 20, 50, 200, 2000]:
#     print('--- r_0=%d ---' % r_0)
#     log,net = train(n_epochs=10, r_0=r_0)
#     logs.append(log)
