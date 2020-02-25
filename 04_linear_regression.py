from torch import nn
import torch
from torch import nn
import torch
from torch import tensor

x_data = tensor([[1.0],[2.0],[3.0]])
y_data = tensor([[2.0],[4.0],[6.0]])

class Model(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__() ## 부모클래스를 호출할때 super를 사용 즉 nn.Module 의 __init__ 메소드 호춣
        self.linear = torch.nn.Linear(1,1) # One in and One out

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return a Variable of
        output data. we can use modules defined in the constructor as well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred

# our model
model = Model()

# Construct our loss function and an Optimizer. the call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum') # criterion : 표준
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model.forward(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()}')

    # 3) Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("prediction (after training)", 4, model(hour_var).data[0][0].item())
