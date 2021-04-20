#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchsummary import summary

import time
import random

import numpy as np
import matplotlib.pyplot as plt


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# In[3]:


#### FUNCTIONS DECLARATION ###


# In[4]:


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) #kernel = filter size. #out = number of filters
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
      
    
    def forward(self, t):   
    #hidden conv layers
        t = self.conv1(t) 
        t = F.relu(t) #activation function
        t = F.max_pool2d(t, kernel_size=2, stride=2)
    
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
    
    #hidden linear layers.
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
    
        t = self.fc2(t)
        t = F.relu(t)
    
    #output layer
        t = self.out(t)
  
        return t


# In[5]:


def get_num_correct(preds, labels): 
    return preds.argmax(dim=1).eq(labels).sum().item()


# In[6]:


def training(network, loader, optimizer, num_epochs):
    for epoch in range(num_epochs):

        total_loss = 0
        total_correct = 0

        for batch in loader: #così prendo tutti i batch e quindi il dataset completo
            images, labels = batch
    
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
    
            optimizer.zero_grad() #gradient must be reset every time, otherwise it's added to the previous one
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)
    
        print(f'epoch: {epoch}, total_correct: {total_correct}, loss: {total_loss}')


# In[7]:


def testing(network, dataset, loader):
    total_correct = 0
    for batch in loader:
        images, labels = batch
        predictions = network(images)
        correct = get_num_correct(predictions, labels)
        total_correct += correct
    return(f'total correct: {total_correct} / {len(dataset)}. Accuracy: {(total_correct/len(dataset))*100}')


# In[8]:


def find_indices(idx):
    indices = []
    for i in range(len(idx)):
        if idx[i].item() == True:
            indices.append(i)
    return indices


def split_dataset(dataset):
    subsets = []
    for i in range(10):
        idx = mnist.targets==i
        indices = find_indices(idx)
        subset = torch.utils.data.Subset(dataset, indices)
        #print('subset:', i, 'len: ', len(subset))
        subsets.append(subset)
    return subsets


# In[9]:


def example_replay(N, network, memory_dataset, train_dataset, memory_loader, train_loader):
    crumbs = []
    for digit in digits:
        l = len(digit)
        indices = random.sample(range(1,l), N)
        crumb = torch.utils.data.Subset(digit, indices)
        crumbs.append(crumb)
    crumbs.append(train_dataset)
    dirty_dataset = torch.utils.data.ConcatDataset(crumbs)
    dirty_loader = torch.utils.data.DataLoader(dirty_dataset, batch_size=100, shuffle=True)
    print(f'Sto rinfrescando la memoria con {N} elementi da mnist per ogni classe. dirty: {len(dirty_dataset)}')
    opt_replay = optim.Adam(network.parameters(), lr=0.01) 
    training(network, dirty_loader, opt_replay, 10)
    print(f'Results for mnist: {testing(network, memory_dataset, memory_loader)}') 
    print(f'Results for training-dataset: {testing(network, train_dataset, train_loader)}')
    print('    ')


# In[10]:


#### DATASETS ###


# In[11]:


USPS_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


# In[12]:


mnist = torchvision.datasets.MNIST(
                        root='./data'
                       ,train=True
                       ,download=True
                       ,transform = transforms.Compose([transforms.ToTensor()])
                        )

fashion = torchvision.datasets.FashionMNIST(
                        root='./data/FashionMNIST'
                        ,train=True
                        ,download=True
                        ,transform=transforms.Compose([transforms.ToTensor()])
                        )

usps = torchvision.datasets.USPS("./data"
                     , train=True
                     , download=True
                     , transform = USPS_transform
                    )


# In[13]:


mnist_loader = torch.utils.data.DataLoader(mnist, batch_size = 100, shuffle=True)
fashion_loader = torch.utils.data.DataLoader(fashion, batch_size = 100, shuffle=True)
usps_loader = torch.utils.data.DataLoader(usps, batch_size = 100, shuffle=True)


# In[14]:


######## CREIAMO IL NOSTRO NETWORK #####

network = Network()
optimizer = optim.Adam(network.parameters(), lr=0.01)


# In[15]:


######## FACCIAMO IL TRAINING SU MNIST #######

training(network, mnist_loader, optimizer, 10)


# In[16]:


#### SALVIAMO QUESTA RETE #####

torch.save(network.state_dict(), 'PATHS/mnist_trained.pth')


# In[17]:


#### FACCIAMO IL TEST SU MNIST/USPS/FASHION

print('mnist: ', (testing(network, mnist, mnist_loader)))
print('usps: ', (testing(network, usps, usps_loader)))
print('fashion: ', (testing(network, fashion, fashion_loader)))


# In[21]:


#### FACCIAMO IL TRAINING SU USPS SENZA MEMORIA ###

training(network, usps_loader, optimizer, 10)


# In[22]:


### FACCIAMO IL TEST SU MNIST/USPS ###
print('mnist: ', (testing(network, mnist, mnist_loader)))
print('usps: ', (testing(network, usps, usps_loader)))


# In[24]:


### RIPRENDIAMO LA RETE MEMORIZZATA AL SOLO MNIST TRAINED E PROVIAMO A FARE IL TRAINING SU USPS CON MEMORIA DI MNIST ###
### il nostro obiettivo sarà quello di migliorare l'accuratezza del mnist dell' 85% ###
digits = split_dataset(mnist)
network_mnist = Network()
for N in (1,2,5,10,50,100,500,1000,2000,5000):
    network_mnist.load_state_dict(torch.load('PATHS/mnist_trained.pth'))
    example_replay(N, network_mnist, mnist, usps, mnist_loader, usps_loader)


# In[25]:


########################################################################################################


# In[26]:


# PROVIAMO A FARE LA STESSA COSA CON IL FASHION ####


# In[28]:


network_mnist.load_state_dict(torch.load('PATHS/mnist_trained.pth'))
optimiz = optim.Adam(network_mnist.parameters(), lr=0.01)
training(network_mnist, fashion_loader, optimiz, 10)


# In[30]:


print('mnist: ', (testing(network_mnist, mnist, mnist_loader)))
print('fashion: ', (testing(network_mnist, fashion, fashion_loader)))


# In[31]:


#qui si vede meglio che il mnist è peggiorato tantissimo!! se facciamo l'example replay come migliorerà?


# In[32]:


for N in (1,2,5,10,50,100,500,1000,2000,5000):
    network_mnist.load_state_dict(torch.load('PATHS/mnist_trained.pth'))
    example_replay(N, network_mnist, mnist, fashion, mnist_loader, fashion_loader)


# In[ ]:




