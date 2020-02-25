from torch.utils.data import Dataset, DataLoader
from torch import nn, from_numpy, optim
import numpy as np

class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./diabetes.csv', delimiter=',', dtype= np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:,0:-1])
        self.y_data = from_numpy(xy[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset = dataset,
                          batch_size= 32,
                          shuffle= True,
                          num_workers = 0)


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1 = nn.Linear(8,6)
        self.l2 = nn.Linear(6,4)
        self.l3 = nn.Linear(4,1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

model = Model()
"""
BCELoss 함수를 사용할 경우에는 먼저 마지막 레이어 값이 0~1로 되어야 하고
마지막 레이어를 시그모이드함수를 적용시켜줘야 한다.
"""
criterion = nn.BCELoss(reduction='sum')
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
if __name__ ==  '__main__':
    for epoch in range(2):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            print(inputs.data)

            y_pred = model(inputs)

            loss = criterion(y_pred, labels)
            print(f'Epoch {epoch + 1} | Batch: {i + 1} | Loss: {loss.item():.4f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



