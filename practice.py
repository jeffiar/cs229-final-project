import torch
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 32

transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define network architecture

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

        # TODO how to initialize this network in a way you want??

    def forward(self, x):
        # And here's how the architecture is pieced together
        # Seems like it's fine practice to just call the "current state"
        # as you pass forward a mutable x.
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
# net.to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def calc_test_acc():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # images = images.to('cuda')
            # labels = labels.to('cuda')
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct/total 

log = dict()
log['loss'] = []
log['test_acc'] = []

print_every=250
for epoch in range(15):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # inputs = inputs.to('cuda')
        # labels = labels.to('cuda')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % print_every == (print_every - 1):
            log['loss'] += (epoch, i * BATCH_SIZE, 
                            running_loss / print_every)
            running_loss = 0.0

    test_acc = calc_test_acc()
    log['test_acc'] = (epoch, test_acc)

    print('Epoch %d. Loss: %.3f, Test accuracy: %.2f%%' % 
            (epoch + 1, running_loss / print_every, test_acc * 100))

print('Finished Training')
