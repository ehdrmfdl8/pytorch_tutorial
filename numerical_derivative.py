import numpy as np
from datetime import datetime

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) #f(x+delta_x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x) #f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2 * delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad

class Diabetes:
    def __init__(self, name, xdata,tdata,i_node, h1_node,o_node,learning_rate, iteration_count):
        self.name = name
        self.xdata = xdata
        if xdata.ndim == 1:
            self.xdata = xdata.reshape(-1,1)
            self.tdata = tdata.reshape(-1,1)
        elif xdata.ndim == 2:
            self.xdata = xdata
            self.tdata = tdata.reshape(-1,1)

        self.i_node = i_node
        self.h1_node = h1_node
        self.o_node = o_node
        self.learning_rate = learning_rate
        self.iteration_count = iteration_count

        self.W2 = np.random.rand(self.i_node, self.h1_node)
        self.b2 = np.random.rand(self.h1_node)

        self.W3 = np.random.rand(self.h1_node, self.o_node)
        self.b3 = np.random.rand(self.o_node)

    def loss_func(self):
        delta = 1e-7

        z2 = np.dot(self.xdata, self.W2) + self.b2
        a2 = sigmoid(z2)

        z3 = np.dot(a2,self.W3) + self.b3
        y = a3 = sigmoid(z3)

        return -np.sum(self.tdata * np.log(y+delta) + (1-self.tdata) * np.log(1-y +delta))

    def predict(self, test_xdata):
        z2 = np.dot(test_xdata, self.W2) + self.b2
        a2 = sigmoid(z2)

        z3 = np.dot(a2,self.W3) + self.b3
        y = a3 = sigmoid(z3)

        if y >= 0.5:
            result = 1
        else:
            result = 0

        return y,result

    def accuracy(self, test_xdata, test_tdata):
        matched_list = []
        not_matched_list = []
        index_label_prediction_list = []
        for index in range(len(test_xdata)):
            (real_val, logical_val) = self.predict(test_xdata[index])
            if logical_val == test_tdata[index]:
                matched_list.append(index)
            else:
                not_matched_list.append(index)
            index_label_prediction_list.append([index, test_tdata[index][0], logical_val])
        print("accuracy = >", len(matched_list) / len(test_xdata))
        return matched_list, not_matched_list, index_label_prediction_list

    def train(self):
        f = lambda x : self.loss_func()
        start_time = datetime.now()
        for step in range(self.iteration_count):
            self.W2 -= self.learning_rate * numerical_derivative(f, self.W2)
            self.b2 -= self.learning_rate * numerical_derivative(f, self.b2)

            self.W3 -= self.learning_rate * numerical_derivative(f, self.W3)
            self.b3 -= self.learning_rate * numerical_derivative(f, self.b3)

            if(step % 400) == 0:
                print("step = ", step, "error_val = ", self.loss_func())
        end_time = datetime.now()
        print("")
        print("elasped time: ", end_time - start_time)

load_data = np.loadtxt('./diabetes.csv',delimiter=',', dtype = np.float32)
x_data = load_data[0:500,0:-1]
t_data = load_data[0:500,[-1]]

test_x_data = load_data[501:,0:-1]
test_t_data = load_data[501:,[-1]].reshape(-1,1)

#데이터 차원 및 shape 확인
print("x_data.ndim = ", x_data.ndim,", x_data.shape = ", x_data.shape)
print("t_data.ndim = ", t_data.ndim,", t_data.shape = ", t_data.shape)

i_node = x_data.shape[1]
h1_node = 2
o_node = t_data.shape[1]

lr = 1e-3
iter_count = 10001

obj = Diabetes("Diabetes",x_data,t_data,i_node,h1_node,o_node,lr,iter_count)
obj.train()
(matched_list,not_not_matched_list, index_label_prediction_list) = obj.accuracy(test_x_data,test_t_data)
print(index_label_prediction_list)