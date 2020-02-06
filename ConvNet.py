from Build_read_data import DogsVsCats as dc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm as tqdm
import os
#need to have the dogs vs cats data in a folder
get_data = dc()
data = get_data.loadSavedData()


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        '''
        32 = convolutional features
        5 = kernel size of 5x5 window as it searches features
        '''
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        '''
        The -1 tells pytorch to expect data of any size
        The 1 refereces the starting data in nn.Conv2d(1, 32, 5)
        one image coming in
        '''
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)
        '''
        first Linnear layer requires the input data shape and outputs to 512
        this unflattened data has a unknown shape so it is calculated by convs(x)
        X represents random data with the same shape as our image data
        print(data[0][0].shape)  = (50,50) 
        '''
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 2x2 refs the kernel
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        if self._to_linear is None:
            '''
            0th references the first batch of data to enter
            this grabs the shape of the data
            '''
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #dims x is a batch thus dim 0 is all of the batches
        #dim 1 references the objects
        return F.softmax(x, dim=1)


net = Net()

optimiser = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

#this gets the image in the right format
X = torch.tensor([i[0] for i in data]).view(-1, 50, 50)

#scale for effeciency gets pixels between 0 and 1
X = X/255

#this gives the labels
y = torch.tensor([i[1] for i in data])

#state value percent to test again i.e 10% in this case
VAL_PCT = 0.1
#this is the validation size
val_size = int(len(X) * VAL_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]


use_cuda = torch.cuda.is_available()
#move device to gpu
device = torch.device('cuda:0' if use_cuda else 'cpu')

net = net.to(device)
train_X = train_X.to(device=device, dtype=torch.float)
train_y = train_y.to(device=device, dtype=torch.float)
test_X = test_X.to(device=device, dtype=torch.float)
test_y = test_y.to(device=device, dtype=torch.float)


batchSize = 100

epoch = 2


#training here
print("training")
for event in range(epoch):
    #range(0) = start at 0
    #len(number in training data)
    for i in tqdm.tqdm(range(0, len(train_X), batchSize)):
        batch_X = train_X[i:i+batchSize].view(-1, 1, 50, 50)
        batch_y = train_y[i:i + batchSize]
        #zero grad one each iteration
        #can do optimiser.grad for different optimisers
        net.zero_grad()
        outputs = net(batch_X)
        #define loss
        loss = loss_function(outputs, batch_y)
        #back propagation
        loss.backward()
        optimiser.step() # Does the update

correct = 0
total = 0

with torch.no_grad():
    for i in range(len(test_X)):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
        predictedClass = torch.argmax(net_out)
        #print(predictedClass, real_class)
        if predictedClass == real_class:
            correct += 1
        total +=1
    accuracy = round(correct/total, 3)
    print("accuracy: {} ".format(accuracy))

cwd = os.getcwd()

torch.save(net.state_dict(), os.path.join(cwd, "\\trainedData"))

