'''
CNN implementation with pytorch, with MNIST dataset
'''

from __future__ import print_function
from model import CNN
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

##### Setting up hyper parameters
batch_size = 100
lr = 0.1
mom = 0.5
PATH = '/home/wataru/Uni/spring_2018/4997/CSCI4997_machine_learning/CSCI4997_CNN/model/save'


##### Import Dataset #####

# Load dataset and create dataloaders

train_dataset = datasets.MNIST(root = './data/', train=True,
                                transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(root = './data/', train=False,
                                transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
##### Define model and Optimizer #####
model = CNN()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
criterion = nn.CrossEntropyLoss()

##### Training #####
def train(epoch):
    model.train() ##what for??
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        #print(target.shape)
        optimizer.zero_grad()
        output = model(data)
        print("output:", output.shape)
        print('target: ', target.shape)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))
 
                    
def test():
    model.eval() ##??
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        #print('shape of target:', target.shape)
        test_loss += criterion(output, target).data[0]
        # get the index of teh max
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
for epoch in range(1, 3):
    train(epoch)
    test()
print('saving model...')
torch.save(model.state_dict(), PATH)
                    
