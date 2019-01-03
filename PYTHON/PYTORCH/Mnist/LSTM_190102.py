import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='/home/qilong/DATA/Mnist_B/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='/home/qilong/DATA/Mnist_B/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  #  hiddenLayer 是 128
        self.num_layers = num_layers  # 2层Layer
        #                       x=28       h=c=128       2层             batch的大小放在前面。
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #                       128行       10类
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 这里h0和c0的大小和隐含层是一样的，都是128，如果不写默认是0,可以不写
        # 注意：这里 X 三维矩阵： 第一维是batch size, 第二维是序列长度，第三维是单个序列矩阵长度
        #      这里 h0和c0是三维矩阵，第一维是LSTM层数，第二维是batch size,第三维是隐含层长度
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM，这里(h0,c0)可以不写，默认为0.
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        
        # 注意：这里 out 三维矩阵： 第一维是batch size, 第二维是序列长度，第三维是单个序列矩阵长度
        # 下面是取每一个batch的最后一个序列的输出序列（因为每一个seq都会输出一个序列） 
        out = self.fc(out[:, -1, :])
        return out

#               28,         128,          2层，        10类
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model 一共多少张图片
total_step = len(train_loader)
# 2遍
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 这里是MNIST数据集，因为本身是两维：第一维是batch_size,第二维是单个序列长度即28×28
        # 这里对它reshape,变成三维：第一维是batch_size,第二维是seq个数，第三维是单个seq长度，正好符合上面函数需要喂入的数据
        images = images.reshape(-1, sequence_length, input_size).to(device)
        # 加载lables
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 