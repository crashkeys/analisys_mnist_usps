import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from colorama import Fore, Style

import math


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


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def training(network, loader, optimizer, num_epochs):
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
    for batch in loader:
        images, labels = batch
        predictions = network(images)
        correct = get_num_correct(predictions, labels)
        total_correct += correct
    return(f'total correct: {total_correct} / {len(dataset)}. {Fore.LIGHTMAGENTA_EX}Accuracy: {(total_correct/len(dataset))*100}{Style.RESET_ALL}')


def temp_scale(Y, T): #prede un vettore (immagine) e ridà il vettore scalato
    Y = F.softmax(Y, dim=0)
    Y_new = []
    for y in Y:
        y_new = ((y.item())**(1/T)) / ((sum(Y).item()**(1/T)))
        Y_new.append(y_new)
    return torch.tensor(Y_new)


def knowledge_distillation(preds, target): #per una singola immagine
    result = 0
    for j in range(10):
        result += temp_scale(target, 2)[j] * math.log(temp_scale(preds, 2)[j] + 1e-34)
    return -result


def know_dist_batch(preds, target):
    V = []
    for idx in range(len(preds)):
        kn = knowledge_distillation(preds[idx], target[idx])
        V.append(kn)
    return sum(V) / len(V)


def get_one_hot(target,num_class):    #ritorna un vettore probabilità dal label
    one_hot = torch.zeros(target.shape[0], num_class)
    one_hot = one_hot.scatter(dim=1, index=target.long().view(-1,1), value=1.)
    return one_hot


def LwF(network_new, network_old, loader, opt, lam, num_epochs):
    for epoch in range(num_epochs):

        total_loss = 0
        total_correct = 0

        for batch in loader:
            images, labels = batch

            preds_new = network_new(images)  # new network, new images
            preds_old = network_old(images)  # old network, new images

            loss_new = F.cross_entropy(preds_new, labels) # semplice fine-tuning

            loss_old = know_dist_batch(preds_new, preds_old)  # mitigate forgetting

            loss = loss_new + (lam * loss_old)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            total_correct += get_num_correct(preds_new, labels)
        print(f'epoch: {epoch}, total_correct: {total_correct}, loss: {total_loss}')
