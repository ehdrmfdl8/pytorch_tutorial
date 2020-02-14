import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0
#our model forward pass
def forward(x):
    return x * w

# Loss function
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# compute gradient
def gradient(x,y):
    return 2* x * (x * w - y)

# before training
print("prediction (before training)", 4, forward(4))

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        # compute derivative w.r.t to the learned weights
        # Update the Weights
        # Compute the loss and print progress
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad
        print("\t grad : ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
    print("progress:", epoch, "w=", round(w,2), "loss=", round(l,2))

# After training
print("predicted score (after training)", "4 hours of studying: ", forward(4))