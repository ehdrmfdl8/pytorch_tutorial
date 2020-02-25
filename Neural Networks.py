"""
신경망은 torch.nn 패키지를 사용하여 생성할수 있습니다.
nn은 모델을 정의하고 미분하는데 autograd를 사용합니다. nn.Module은 계층(layer)과 output을 반환하는
forward(input) 메서드를 포함하고 있습니다.

신경망의 일반적인 학습과정은 다음과 같습니다.
- 학습 가능한 매개변수(또는 가중치(weight))를 갖는 신경망을 정의합니다. -> self.linear = torch.nn.Linear(1,1)
- 데이터셋(dataset) 입력을 반복합니다. -> for epoch in range(500):
- 입력을 신경망에서 전파(process)합니다.-> y_pred = model.forward(x_data)
- 손실(loss; 출력이 정답으로 부터 얼마나 떨어져 있는지)을 계산합니다.-> loss = torch.nn.MSELoss(reduction='sum')
- 변화도(gradient)를 신경망의 매개변수들에 역으로 전파합니다. -> loss.backword()
- 신경망의 가중치를 갱신합니다. 일반적으로 다음과 같은 간단한 규칙을 사용합니다. -> optimizer.step()
        가중치(w) = 가중치(w) - 학습률(learning rate) * 변화도(gradient)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3 x 3 square convolution
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
