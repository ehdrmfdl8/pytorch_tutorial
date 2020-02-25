# PyTorch 라이브러리 임포트

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Scikit-Learn 라이브러리 임포트
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Pandas 라이브러리 임포트
import pandas as pd

wine = load_wine()
wine
# 데이터 프레임에 담긴 설명변수 출력
pd.DataFrame(wine.data, columns=wine.feature_names)
wine.target

wine_data = wine.data[0:130]
wine_target = wine.target[0:130]
#데이터 집합을 훈련 데이터와 테스트 데이터로 분할
train_X, test_X, train_Y, test_Y = train_test_split(wine_data,wine_target,test_size=0.2)

#데이터 건수 확인
print(len(train_X))
print(len(test_X))
print(type(train_X))

# 훈련 데이터 텐서 변환
train_x = torch.from_numpy(train_X).float()
train_y = torch.from_numpy(train_Y).long()

#테스트 데이터 텐서 변환
test_x = torch.from_numpy(test_X).float()
test_y = torch.from_numpy(test_Y).long()
#텐서로 변환한 데이터 건수 확인
print(train_x.shape)
print(train_y.shape)

# 설명 변수와 목적 변수의 텐서를 합침
train = TensorDataset(train_x, train_y)

#텐서의 첫 번째 데이터 내용 확인
print(train[0])

#미니 배치로 분할
train_loader = DataLoader(train, batch_size= 16, shuffle= True)
print(train_loader)


# 신경망 구성
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(13, 96)
        self.l2 = nn.Linear(96, 2)

    def forward(self, x):
        out1 = F.relu(self.l1(x))
        out2 = self.l2(out1)
        y_pred =  F.log_softmax(out2)
        return y_pred


# 인스턴스 생성
model = Net()
# 오차함수 객체
criterion = nn.CrossEntropyLoss()

# 최적화를 담당할 객체
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 학습 시작
for epoch in range(300):
    total_loss = 0
    # 분할해 둔 데이터를 꺼내옴
    for train_x, train_y in train_loader:
        # 계산 그래프 구성
        train_x, train_y = Variable(train_x), Variable(train_y)
        # 경사 초기화
        optimizer.zero_grad()
        # 순전파 계산
        output = model.forward(train_x)
        # 오차 계산
        loss = criterion(output, train_y)
        # 역전파 계산
        loss.backward()
        # 가중치 업데이트
        optimizer.step()
        # 누적 오차 계산
        total_loss += loss.item()
    
    if (epoch + 1) % 50 == 0:
        print(epoch + 1, total_loss)
