'''
Simple linnear neural network where we define layers to guess MNIST. As is this will train and predict. 
We can see loss go down and accuracy increase. Also this trains on the gpu. 
Still need to figure out how to save these and also how to use established models instead of 
defining our own architecture. Starting with this because I'll understand it more
'''
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt

'''
from pytorch docs
A typical training procedure for a neural network is as follows:

Define the neural network that has some learnable parameters (or weights)
Iterate over a dataset of inputs
Process input through the network
Compute the loss (how far is the output from being correct)
Propagate gradients back into the network’s parameters
Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient
'''

#this loads MNIST and needs to be transformed to a tensor.
train = datasets.MNIST('', train=True,
                       download=True,
                       transform=transforms.ToTensor())

test = datasets.MNIST('', train=False,
                      download=True,
                      transform=transforms.ToTensor())

trainSet = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testSet = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)


#some important vars
learningRate=0.001
EPOCH=3

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #input and output to each layer
        #input is 28x28 / 784 because images are 28x28
        #double check with .shape
        #output is 64 just cause, any number can work
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        return f.log_softmax(x, dim=1)


net = Net()
#prints true
use_cuda = torch.cuda.is_available()
#move device to gpu
device = torch.device('cuda:0' if use_cuda else 'cpu')
net = net.to(device)

'''
Adam optimiser
learning rate takes steps
Best is to have large steps that decay over time
'''
optimiser = optim.Adam(net.parameters(), lr=learningRate)

for epoch in range(EPOCH):
    print("currently at epoch: {}".format(epoch))
    for data in trainSet:
        X, y = data
        #NB also need to move data to gpu
        X = X.to(device)
        y = y.to(device)
        '''
        data cointains a tensor of the image and labels as a tensor of [0:9]
        data sent in as batches of 10 see the data loader func
        batches help generalisation and keeping things small
        print(X[0].shape)
        print(y[0].shape)
        '''

        #zero_grad because we batches, this is to continue to accurately calc loss
        net.zero_grad()
        #view is a reshaping function, not sure if its neccesary as data is in right format?
        output = net(X.view(-1, 28 * 28))
        #for one hot vectors use MSE for loss
        #For scalar vals use nll_loss
        loss = f.nll_loss(output, y)
        #back propagate the loss
        loss.backward()
        #adjust weights
        optimiser.step()

    print(loss)


'''
With no_grad means we dont adjust anything 
We just use the model to predict
similiar to .eval mode
'''
correct = 0
total = 0
with torch.no_grad():
    for data in trainSet:
        X, y = data
        X = X.to(device)
        y = y.to(device)
        output = net(X.view(-1, 28 * 28))
        '''
        Neural net outputs argmax
        '''
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1

        print("Accuracy", round(correct / total, 3))




