import numpy as np


# define the activation function
def f(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    shift_x = x - np.max(x)    # 防止输入增大时输出为nan
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)


# derivative of activation  function
# sigmoid导数
def df(x):
    return f(x) * (1-f(x))

# ReLU导数
def reluPrime(x):
    w, h = x.shape
    d = np.zeros((w, h))
    d[x <= 0] = 0
    d[x >= 0] = 1
    return d


# Step 4: Define Cost Function
# MSE
def cost(a, y):
    J = 1/2 * np.sum((a - y)**2)
    return J

# 交叉熵
def cross_entropy(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))


# Step 5: Define Evaluation Index
def accuracy(a, y):
    mini_batch = a.shape[1]
    idx_a = np.argmax(a, axis=0)
    idx_y = np.argmax(y, axis=0)
    acc = sum(idx_a == idx_y) / mini_batch
    return acc


# Forward computation function for model 1
def fc(w, a, b):
    z_next = np.dot(w, a) + b
    a_next = f(z_next)
    return a_next, z_next

# Forward computation function for model 2
def fc2(w, a):
    z_next = np.dot(w, a)
    a_next = relu(z_next)
    return a_next, z_next


# Backward computation function for model 1
def bc(w, z, delta_next, beta):
    delta = (np.dot(w.T, delta_next) + beta) * df(z)
    return delta

# Backward computation function for model 2
def bc2(w, z, delta_next, beta):
    delta = (np.dot(w.T, delta_next)) * reluPrime(z)
    return delta
