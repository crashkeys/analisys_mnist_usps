import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from colorama import Fore, Style
import matplotlib.pyplot as plt

import Network


class EWC_obj(object):
    def __init__(self, model: nn.Module):

        self.model = model

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.saved = {}
        self.gradient = self.fisher()

        for n, p in deepcopy(self.params).items():
            self.saved[n] = p.data

    def fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = p.data

        self.model.eval()

        for n, p in self.model.named_parameters():
            precision_matrices[n].data += p.grad.data ** 2

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty_ewc(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self.gradient[n] * (p - self.saved[n]) ** 2
            loss += _loss.sum()
        return loss

    def penalty_l2(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = (p - self.saved[n]) ** 2
            loss += _loss.sum()
        return loss


def ewc_train(network, optimizer, dataset, loader, ewc, lam, num_epochs, type="ewc", test=None):
    for epoch in range(num_epochs):
        total_loss = 0
        total_correct = 0

        for batch in loader:
            images, labels = batch

            preds = network(images)
            loss1 = F.cross_entropy(preds, labels)

            if type == "L2":
                loss2 = lam * ewc.penalty_l2(network)
            else:
                loss2 = lam * ewc.penalty_ewc(network)

            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += Network.get_num_correct(preds, labels)

        accuracy = (total_correct / len(dataset)) * 100
        print(f'epoch: {epoch}, loss: {total_loss}, total_correct: {total_correct} / {len(dataset)}, --> {Fore.LIGHTCYAN_EX}Accuracy: {accuracy}{Style.RESET_ALL}')
        
        #test is for printing epoch accuracies of past tasks
        if test is not None:
            for t in test:
                print(f"\t\t\t\t {Fore.LIGHTGREEN_EX}Testing back... {Network.testing(network, t[0], t[1])}{Style.RESET_ALL}")



