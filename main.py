'''
CNN implementation with pytorch, with MNIST dataset
'''

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

##### Setting up hyper parameters
batch_size = 100
lr = 0.03
mom = 0.5

##### Import Dataset #####

# Load dataset and create dataloaders

##### Define model and Optimizer #####
model = CNN()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)

##### Training #####
def train(epoch):
    model.train() ##what for??
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))


