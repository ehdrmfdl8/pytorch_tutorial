from __future__ import print_function
from  torch import nn, optim, cuda
from torch.utils import data
from torchvision import datasets, transforms
import torch.nn.functional as F
from datetime import datetime

#트레이닝 설정
batch_size = 64
device = 'cuda' if cuda.is_available() else 'cpu'
print(f'Training MNIST model on {device} \n{"=" * 44}')

#MNIST 데이터 셋
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader
train_loader = data.DataLoader(dataset= train_dataset,
                               batch_size=batch_size,
                               shuffle=True)
test_loader = data.DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=True)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784) # Flatten the data (n, 1, 28, 28) -> (n, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    model.train()
    for batch_idx,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} | Batch Status: {batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%) | Loss: {loss.item():.6f}')

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(f'==============================\nTest set: Average loss : {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f'({100. * correct / len(test_loader.dataset):.0f}%)')

if __name__ == '__main__':
    since = datetime.now()

    for epoch in range(1, 10):
        epoch_start = datetime.now()
        train(epoch)
        print(datetime.now() - epoch_start)
        test()
        print(datetime.now() - epoch_start)

    print(f'Total Time: {datetime.now()- since}\nModel was trained on {device}!')