from torch import nn, tensor, max
import numpy as np

Y = np.array([1, 0, 0])
Y_pred1 = np.array([0.7, 0.2, 0.1])
Y_pred2 = np.array([0.1, 0.3, 0.6])
print(f'Loss1 : {np.sum(-Y * np.log(Y_pred1)):.4f}')
print(f'Loss2 : {np.sum(-Y * np.log(Y_pred2)):.4f}')
# Softmax + CrossEntropy (logsoftmax + NLLLoss)
loss = nn.CrossEntropyLoss()

Y = tensor([0], requires_grad=False)

Y_pred1 = tensor([[2.0, 1.0, 0.1]])
Y_pred2 = tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)

print(f'PyTorch Loss1: {l1.item():.4f} \nPyTorch Loss2: {l2.item():.4f}')
print(f'Y_pred1: {max(Y_pred1.data, 1)[1].item()}')
print(f'Y_pred2: {max(Y_pred2.data, 1)[1].item()}')

Y = tensor([2, 0 ,1], requires_grad=False)

Y_pred1 = tensor([[0.1, 0.2, 0.9],
                  [1.1, 0.1, 0.2],
                  [0.2, 2.1, 0.1]])
Y_pred2 = tensor([[0.8, 0.2, 0.3],
                  [0.2, 0.3, 0.5],
                  [0.2, 0.2, 0.5]])

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)
print(f'Batch Loss1: {l1.item():.4f} \nBatch Loss2: {l2.data:.4f}')
