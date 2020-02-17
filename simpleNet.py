import torch
import torch.utils.data as torchData
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from collections import deque
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

####Controls#####

imageName = "car.jpg"   #feed in a example image use class names
modelName = "cifar_net_trained_dynLearning.pth"

TRAIN = False        #test or train
DYNAMICLEARNING = False
LEARNINGRATE = 0.001
BATCH_SIZE = 10


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


trainset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                transform=transform,
                                                download=True)

trainloader = torchData.DataLoader(trainset,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data',
                                               train=False,
                                               transform=transform,
                                               download=True)

testloader = torchData.DataLoader(testset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def cpu_gpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def optimiser(learining_rate, method, net):
    if method == "sgd" or method == "SGD":
        return optim.SGD(net.parameters(), lr=learining_rate, momentum=0.9)
    elif method == "adam" or method == "Adam":
        return optim.Adam(net.parameters(), lr=learining_rate)


def trainModel(net, loss_function, EPOCH, LR, DynamicLearning):
    ACC = deque([0, 0])
    net = net.to(device=cpu_gpu())
    f_loss = open('loss.txt', "a")
    f_loss.write("Time" + '\t' + "loss" + '\n')

    f_acc = open('acc.txt', "a")
    f_acc.write("Epoch" + '\t' + "Accuracy" + '\n')

    for epoch in range(EPOCH):
        print("Starting {}/{} epochs".format(epoch, EPOCH))
        running_loss = 0
        f_accContent = str(epoch) + '\t' + str(ACC[1]) + '\n'
        f_acc.write(f_accContent)
        if epoch > 0:
            ACC.popleft()

        if epoch >= 1:
            if DynamicLearning:
                step = np.random.randint(low=1, high=10, size=1).item()
                if LR > 0:
                    if ACC[0] < ACC[1]:
                        LR = LR + (LR / step)
                    elif ACC[0] > ACC[1]:
                        LR = LR - (LR / step)
                    elif ACC[0] == ACC[1]:
                        switchAcc = np.random.randint(0, 2, size=10).sum()
                        if switchAcc > 5:
                            LR = LR - np.random.uniform(0, LR, size=1).item()
                        elif switchAcc < 5:
                            LR = LR + np.random.uniform(0, LR, size=1).item()
                        else:
                            LR = np.random.uniform(0, 0.01, size=1).item()
                else:
                    LR = np.random.uniform(0, 0.01, size=1).item()
            else:
                LR = LEARNINGRATE

        print("current accuracy Q: {}, current learning rate {}".format(ACC, LR))

        for i, trainData in enumerate(trainloader, 0):
            images, labels = trainData[0].to(cpu_gpu()), trainData[1].to(cpu_gpu())

            optimiser(LR, method="sgd", net=net).zero_grad()
            output = net(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimiser(LR, method="sgd", net=net).step()

            running_loss += loss.item()


            f_content = str(time.time()) + '\t' + str(loss.item())
            f_loss.write(f_content + '\n')


            if i % 1000 == 999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0

        testcorrect = 0
        testtotal = 0

        with torch.no_grad():
            for testdata in testloader:
                testimages, testlabels = testdata[0].to(cpu_gpu()), testdata[1].to(cpu_gpu())
                testOut = net(testimages)
                _, testpredicted = torch.max(testOut.data, 1)
                testcorrect += (testpredicted == testlabels).sum().item()
                testtotal += testlabels.size(0)

        accuracy = (100 * testcorrect / testtotal)
        ACC.append(accuracy)

    PATH = os.path.join("CIFAR_models", "cifar_net_{}.pth".format(ACC[1]))
    torch.save(net.state_dict(), PATH)
    f_loss.close()
    f_acc.close()
    print("Done training for {}".format(EPOCH))


def useModel(image, net, PATH):
    net.load_state_dict(torch.load(PATH))
    img = cv2.imread(image)
    img = cv2.resize(img, (32, 32))
    img = transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        result = net(img)
    return result




if TRAIN:
    trainModel(net=Net(), loss_function=nn.CrossEntropyLoss(),
               EPOCH=40, LR=LEARNINGRATE, DynamicLearning=DYNAMICLEARNING)

else:
    out = useModel(image=os.path.join("CIFAR_hires", imageName), net=Net(), PATH=os.path.join("CIFAR_models",
                                                                                                modelName))
    _, predicted = torch.max(out, 1)
    print(classes[predicted])





