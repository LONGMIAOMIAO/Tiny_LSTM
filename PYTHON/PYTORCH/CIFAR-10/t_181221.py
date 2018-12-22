import torch
import torchvision
import torchvision.transforms as transforms
import time

# Normalization
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Load Train Data
trainset = torchvision.datasets.CIFAR10(
    root='/home/qilong/DATA/CIFAR-10', 
    train = True, download=True, transform=transform)

# Set Feed Size
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=False, num_workers=4)

# Load Test Data
testset = torchvision.datasets.CIFAR10(
    root='/home/qilong/DATA/CIFAR-10', 
    train = False, download=True, transform=transform)

# Set Feed Size
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=4)

import torch.nn as nn
import torch.nn.functional as F

# Net Start
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding = 2)
        self.conv2 = nn.Conv2d(64, 128, 5, padding = 2)
        self.poolMax = nn.MaxPool2d(2, 2)
        self.poolAv = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 128, 5, padding = 2)
        self.conv4 = nn.Conv2d(128, 128, 5, padding = 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)
        self.bn64 = nn.BatchNorm2d(64)
        self.bn128 = nn.BatchNorm2d(128)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.05)

    def forward(self, x):
        ## CUDA
        x = x.to(device)
        
        x = self.bn64(F.relu(self.conv1(x)))
        x = self.bn128(F.relu(self.conv2(x)))
        x = self.poolMax(x)
        x = self.dropout10(x)
        
        x = self.bn128(F.relu(self.conv3(x)))
        x = self.bn128(F.relu(self.conv4(x)))
        x = self.poolAv(x)
        x = self.dropout10(x)

        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# creating Net
net = Net()

## CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

import torch.optim as optim
# Loss Function
criterion = nn.CrossEntropyLoss()
# upW Class
# optimizer = optim.SGD(net.parameters(), lr=0.02, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train Start
t0 = time.clock()
for epoch in range(10):  # loop over the dataset multiple times
    totalNUm = 0
    # Loss
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        totalNUm+=1
        # Feed train Data
        inputs, labels = data
        
        ## CUDA
        inputs, labels = inputs.to(device), labels.to(device)
        
        # clear data grad()
        optimizer.zero_grad()
        # forward
        outputs = net(inputs)
        # cal loss
        loss = criterion(outputs, labels)
        # backward
        loss.backward()
        # udate W
        optimizer.step()
        
        # loss tobe print
        running_loss += loss.item()
        if i % 1000 == 999:    
            # print every 1000 mini-batches
            print ('[%d, %5d] loss: %.3f' %( epoch+1, i+1, running_loss / 1000))
            running_loss = 0.0
    print('TotalNUm::%d'%totalNUm)
print('Finished Training')

# cal time
t1 = time.clock()
# print time
print (t1 - t0)

# Test
correct = 0
total = 0
# cancle backward and updateW
with torch.no_grad():
    for data in testloader:
        # feed testData
        images, labels = data

        ## CUDA
        images, labels = images.to(device), labels.to(device)
        
        # calforward
        outputs = net(images)

        # cal correct value
        _, predicted = torch.max(outputs.data, 1)
        # cal totalNum
        total += labels.size(0)
        # cal correct num
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
print('TotalNUm::%d'%total)
print('Correct::%d'%correct)