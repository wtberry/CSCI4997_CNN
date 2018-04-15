'''
CNN model using pytorch
Thanks to pytorch CNN tutorial: https://www.youtube.com/watch?v=LgFNRIFxuUo
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Defining layers
        self.conv1 = nn.Conv2d(1, 10, kernel_sizes=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_sizes=5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        print('size of x:', x.shape)
        input_size = x.size(0)
        xc1 = self.conv1(x)
        print('size of xc1:', xc1.shape)
        xmp1 = self.mp(xc1)
        print('size of xmp1:', xmp1.shape)
        x_relu1 = F.relu(xmp1)

        xc2 = self.conv2(x_relu1)
        print('size of xc2:', xc2.shape)
        xmp2 = self.mp(xc2)
        print('size of xmp2:', xmp2.shape)
        x_relu2 = F.relu(xmp2)
        xfc = self.fc(x_relu2)
        print('size of fc:', xfc.shape)
        # softmax returns probability of each candidates 
        return F.log_softmax(x)

