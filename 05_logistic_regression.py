from torch import tensor
from torch import nn
from torch import sigmoid
import torch.nn.functional as F
import torch.optim as optim

# Training data and ground truth
x_data = tensor([[1.0],[2.0],[3.0],[4.0]])
y_data = tensor([[0.],[0.],[1.],[1.]])

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = sigmoid(self.linear(x))
        return y_pred

# our model
model = Model()

criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training loop
for epoch in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # compute and print loss
    loss = criterion(y_pred,y_data)
    print(f'Epoch {epoch + 1}/ 1000 | Loss: {loss.item():.4f}')

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#After Training
print(f'\nLet\'s predict the hours need to score above 50%\n{"=" * 50}')
hour_var = model(tensor([[1.0]]))
print(f'prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_var = model(tensor([[7.0]]))
print(f'prediction after 7 hours of training: {hour_var.item() : .4f}| Above 50%: {hour_var.item() > 0.5}')