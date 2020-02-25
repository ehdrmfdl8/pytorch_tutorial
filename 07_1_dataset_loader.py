# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, tensor
import numpy as np

class DiabetesDataset(Dataset):
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./diabetes.csv',
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)


"""
본인이 구현한 코드가 다른 파이썬 코드에 의해서 모듈로 import 될 경우도 있고, 파이썬 인터프리터에 의해서 직접
실행될 경우도 있을 수 있는데, 위 코드는 인터프리터에 의해서 직접 실행될 경우에만 실행하도록 하고 싶은 블럭이 있는 
경우에만 사용한다.
"""
if __name__ ==  '__main__': ## 만약 이 파일이 인터프리터에 의해서 실행되는 경우라면
    for epoch in range(2):
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            input, label = data
            # wrap them in Variable
            input, label = tensor(input), tensor(label)

            # Run your training process
            print(f'Epoch: {i} | Inputs {input.data} | Labels {label.data}')
else:
    print("error")