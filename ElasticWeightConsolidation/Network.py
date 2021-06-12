import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


def training(network, dataset, loader, lr, num_epochs):
    accs = []
    optimizer = optim.Adam(network.parameters(), lr=lr)
    for epoch in range(num_epochs):

        total_loss = 0
        total_correct = 0

        for batch in loader:
            images, labels = batch

            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()  # gradient must be reset every time, otherwise it's added to the previous one
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)

        accuracy = (total_correct / len(dataset)) * 100
        accs.append(accuracy)
        print(f'epoch: {epoch}, loss: {total_loss}, total_correct: {total_correct} / {len(dataset)}, --> {Fore.LIGHTMAGENTA_EX}Accuracy: {accuracy}{Style.RESET_ALL}')
    x = [i for i in range(num_epochs)]
    plt.plot(x, accs, color='deeppink', marker='o')


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


