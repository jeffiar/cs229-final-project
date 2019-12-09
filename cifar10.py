import torch
import torchvision
import torchvision.transforms as transforms

BATCH_SIZE = 32

transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

def get_trainloader(batch_size=BATCH_SIZE):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
            shuffle=True, num_workers=2)
    return trainloader

def get_testloader(batch_size=BATCH_SIZE):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
            shuffle=False, num_workers=2)
    return testloader

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
