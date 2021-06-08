
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from copy import deepcopy
from colorama import Fore, Style

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6,
                               kernel_size=5)  # kernel = filter size. #out = number of filters
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # hidden conv layers
        t = self.conv1(t)
        t = F.relu(t)  # activation function
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # hidden linear layers.
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        # output layer
        t = self.out(t)

        return t


def training(network, loader, lr, num_epochs):
    optimizer = optim.Adam(network.parameters(), lr=lr)
    for epoch in range(num_epochs):

        total_loss = 0
        total_correct = 0

        for batch in loader:
            images, labels = batch

            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()  # gradient must be reset every time, otherwise it's added to the previous one
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        print(f'epoch: {epoch}, total_correct: {total_correct}, loss: {total_loss}')


def testing(network, dataset, loader):
    total_correct = 0
    with torch.no_grad():
        for batch in loader:
            images, labels = batch
            predictions = network(images)
            correct = get_num_correct(predictions, labels)
            total_correct += correct
        return (
            f'total correct: {total_correct} / {len(dataset)}. {Fore.LIGHTMAGENTA_EX}Accuracy: {(total_correct / len(dataset)) * 100}{Style.RESET_ALL}')


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_params(model):
    params = {name: param for name, param in model.named_parameters() if param.requires_grad} #dictionary
    return params


def copy_params(model):
    params = get_params(model)
    means = {}
    for n, p in deepcopy(params).items():
        means[n] = p.data
    return means


def penalty(model, saved):  #per ora senza fisher
    loss = 0
    for n, p in model.named_parameters():
        _loss = (p - saved[n]) ** 2
        loss += _loss.sum()
    return loss


def EWC(network, loader, lr, lam,
        num_epochs):  # per ora è senza fisher matrix, dovrebbe fare come l2 (ricorda bene old tasks ma non impara new tasks)
    param_old = copy_params(network)
    opt = optim.Adam(network.parameters(), lr=lr)
    for epoch in range(num_epochs):

        total_loss = 0
        total_correct = 0

        for batch in loader:
            images, labels = batch

            preds = network(images)
            loss_new = F.cross_entropy(preds, labels)  # semplice fine-tuning

            loss_old = (lam / 2) * penalty(network, param_old)  # ewc (senza fisher)

            loss = loss_new + loss_old

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        print(f'epoch: {epoch}, total_correct: {total_correct}, loss: {total_loss}')

########################

mnist = torchvision.datasets.MNIST(root='./data'
                                , train=True
                                , download=True
                                , transform=transforms.Compose([transforms.ToTensor()])
                                )

USPS_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

usps = torchvision.datasets.USPS("./data"
                                     , train=True
                                     , download=True
                                     , transform=USPS_transform
                                     )



SVHN_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])


svhn = torchvision.datasets.SVHN(
                        root='./data'
                        ,split='train'
                        ,transform=SVHN_transform
                        ,download=True)

svhn_loader = torch.utils.data.DataLoader(svhn, batch_size=100, shuffle=True)
mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=100, shuffle=True)
usps_loader = torch.utils.data.DataLoader(usps, batch_size=100, shuffle=True)


## MAIN ##

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = Network()

### TRAINING ON MNIST ###
training(network, mnist_loader, lr=0.01, num_epochs=10)
torch.save(network.state_dict(), 'PATHS/mnist_ewc')
print("\t\t Testing on MNIST: ", testing(network, mnist, mnist_loader))
print("\t\t Testing on SVHN: ", testing(network, svhn, svhn_loader))

### MNIST --> USPS ###
## FINE TUNING ##
for lr in (0.01, 1e-4, 1e-3):
    network.load_state_dict(torch.load('PATHS/mnist_ewc'))
    training(network, usps_loader, lr=lr, num_epochs=10)
    print("\t\t Testing on MNIST: ", testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", testing(network, usps, usps_loader))
torch.save(network.state_dict(), 'PATHS/mnist+usps_ft')

## EWC (L2) ##
for lr in (0.01, 1e-4, 1e-3):
    for lam in (0, 1, 2, 0.1):
        network.load_state_dict(torch.load('PATHS/mnist_ewc'))
        EWC(network, usps_loader, lr=lr, lam=lam, num_epochs=10)
        print("\t\t Testing on MNIST: ", testing(network, mnist, mnist_loader))
        print("\t\t Testing on USPS: ", testing(network, usps, usps_loader))
torch.save(network.state_dict(), 'PATHS/mnist+usps_ewc')


## MNIST + USPS -> SVHN ##
## Fine Tuning ##
for lr in (0.01, 1e-3, 1e-4):
    print("lr --> ", lr)
    network.load_state_dict(torch.load('PATHS/mnist+usps_ft'))
    training(network, svhn_loader, lr=lr, num_epochs=10)
    print("\t\t Testing on MNIST: ", testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", testing(network, svhn, svhn_loader))
    print(" ")


## EWC (L2) ##
for lr in (0.01, 1e-3, 1e-4):
    for lam in (0.1, 1, 2):
        print("lr --> ", lr, " lam --> ", lam)
        network.load_state_dict(torch.load('PATHS/mnist+usps_ewc'))
        EWC(network, svhn_loader, lr=lr, lam=lam, num_epochs=10)
        print("\t\t Testing on MNIST: ", testing(network, mnist, mnist_loader))
        print("\t\t Testing on USPS: ", testing(network, usps, usps_loader))
        print("\t\t Testing on SVHN: ", testing(network, svhn, svhn_loader))
        print(" ")
