import torch
import math
import random
import collections
import itertools
import json
import time
import torch.nn as nn
import torch.nn.functional as F

import nets
import cifar10
import mnist

train_dl = cifar10.get_trainloader()
test_dl = cifar10.get_testloader()
input_dim = 32*32*3

# train_dl = mnist.get_trainloader()
# test_dl = mnist.get_testloader()
# input_dim = 784

def get_acc_and_loss(net, dl):
    correct = 0
    total = 0
    running_loss = 0
    n_batches = 0
    with torch.no_grad():
        for data in dl:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = net(images)
            batch_loss = nn.CrossEntropyLoss()(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            n_batches += 1
            running_loss += batch_loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    loss = running_loss / n_batches
    return acc,loss

def train(n_epochs=10, epoch_start=0, sphere=False,
         Net=nets.FCNet, net=None, log=None, **kwargs):

    if net is None:
        print('Making Network...')
        net = Net(**kwargs)
        net.to('cuda')

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    if log is None:
        log = collections.defaultdict(list)
    print_every = 100
    total_examples_seen = 0
    batch_size = train_dl.batch_size
    n_examples = len(train_dl.dataset.targets)
    batches_per_epoch = 1 + n_examples // batch_size

    i = 0
    i_last_printed = 0
    running_loss = 0
    running_correct = 0
    running_examples_seen = 0

    def do_log(calc_train=False, calc_test=False):
        t = epoch_start + total_examples_seen / n_examples
        to_print = '[t = %.2f] ' % t
        if calc_train:
            train_acc, train_loss = get_acc_and_loss(net, train_dl)
            log['train_acc'].append((t, train_acc))
            log['train_loss'].append((t, train_loss))
        else:
            train_acc = running_correct / running_examples_seen
            train_loss = running_loss / (i - i_last_printed)
            log['train_acc'].append((t, train_acc))
            log['train_loss'].append((t, train_loss))

        to_print += 'Acc: {:.2%} Loss: {:<05.4g}'.format(train_acc, train_loss)

        if calc_test:
            test_acc, test_loss = get_acc_and_loss(net, test_dl)
            log['test_acc'].append((t, test_acc))
            log['test_loss'].append((t, test_loss))
            to_print += '  Val Acc: {:.2%} Loss: {:<05.4g}'.format(test_acc, test_loss)

        r = net.get_radius()
        log['radius'].append((t, r))
        rs = net.get_radii()
        log['radii'].append([t] + rs)
        to_print += '  r: {:g}'.format(r)
        end = '\n' if calc_test or calc_train else '\r'
        print(to_print, end=end)

    print('Measuring initial properties...')
    do_log(calc_train=True)

    print('Starting training...')
    for epoch in range(epoch_start, epoch_start + n_epochs):
        i = 0
        i_last_printed = 0
        running_loss = 0
        running_correct = 0
        running_examples_seen = 0
        for i, data in enumerate(train_dl):
            inputs = data[0].to('cuda')
            labels = data[1].to('cuda')

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            loss.backward()
            optimizer.step()
            if sphere:
                net.weights.data *= r_0 / net.weights.data.norm()

            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            running_examples_seen += len(labels)
            running_loss += loss.item()
            total_examples_seen += len(labels)

            # print statistics
            if i % print_every == (print_every - 1):
                if epoch < 1:
                    do_log(calc_test=True)
                else:
                    do_log(calc_test=False)
                i_last_printed = i
                running_loss = 0
                running_correct = 0
                running_examples_seen = 0
            if i == batches_per_epoch - 1:
                do_log(calc_test=True)

    print('Finished training.')
    return log,net

logs = []
r_0s = [0.2, 0.4, 
        1, 2, 4, 
        10, 20, 40, 
        100, 200, 400, 
        1000, 2000, 4000, 
        10000, 20000, 40000, 
        100000, 200000]
for r_0 in r_0s:
    print('--- r_0 = %.1f ---' % (r_0))
    log, net = train(n_epochs=15, Net=nets.FCNet, sphere=False, input_dim=input_dim, r_0=r_0)
    log['r_0'] = r_0
    log['d'] = net.D
    log['time_finished'] = time.time()
    logs.append(log)

    with open('experiment_3c.json', 'w') as f:
        json.dump(logs, f)


# logs = []
# r_0s = [0.4, 1, 400, 1000, 4000, 10000, 40000, 100000]
# for r_0 in r_0s:
#     # print('--- r_0 = %.1f, d = %d ---' % (r_0, d))
#     log, net = train(n_epochs=15, Net=nets.HypersphereFCNet, sphere=True, input_dim=input_dim, r_0=r_0)
#     log['r_0'] = r_0
#     log['d'] = net.D
#     log['time_finished'] = time.time()
#     logs.append(log)

#     with open('experiment_5b2.json', 'w') as f:
#         json.dump(logs, f)

# # ----

# with open('experiment_4c.json', 'r') as f:
#     logs = json.load(f)

# # logs = []
# r_0s = [0.2, 2, 20, 50, 100, 200, 2000, 20000, 200000]
# ds = [1000, 3000, 7000]
# for d in ds:
#     for r_0 in r_0s:
#         if d == 1000 and r_0 < 200000:
#             continue
#         print('--- r_0 = %.1f, d = %d ---' % (r_0, d))
#         log, net = train(n_epochs=15, Net=nets.SparseRandomHyperplaneNet, sphere=False, input_dim=input_dim, r_0=r_0, d=d)
#         log['r_0'] = r_0
#         log['d'] = d
#         log['time_finished'] = time.time()
#         logs.append(log)

#         with open('experiment_4c.json', 'w') as f:
#             json.dump(logs, f)

# Past runs: need to log these

# r_0s = [2000, 20000, 200000]
# ds = [300, 1000, 3000, 7000, 10000]
# for d in ds:
#     for r_0 in r_0s:
#         print('--- r_0 = %.1f, d = %d ---' % (r_0, d))
#         # log, net = train(n_epochs=15, Net=nets.SparseHypersphereNet, sphere=True, input_dim=input_dim, r_0=r_0, d=d)
#         log, net = train(n_epochs=15, Net=nets.SparseRandomHyperplaneNet, input_dim=input_dim, r_0=r_0, d=d)
#         log['d'] = d
#         log['r_0'] = r_0
#         log['time_finished'] = time.time()
#         logs.append(log)

#         with open('experiment_4b3.json', 'w') as f:
#             json.dump(logs, f)

# logs = []
# r_0s = [0.2, 2, 4, 10, 40, 200, 2000, 20000, 200000]
# ds = [10000, 1000, 1000, 300, 3000]
# for d in ds:
#     if d == 10000 and r_0 < 100:
#         continue
#     for r_0 in r_0s:
#         print('--- r_0 = %.1f, d = %d ---' % (r_0, d))
#         log, net = train(n_epochs=15, Net=nets.SparseHypersphereNet, sphere=True, input_dim=input_dim, r_0=r_0, d=d)
#         log['d'] = d
#         log['r_0'] = r_0
#         log['time_finished'] = time.time()
#         logs.append(log)

#         with open('experiment_6c.json', 'w') as f:
#             json.dump(logs, f)
