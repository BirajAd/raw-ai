import numpy as np

def sgd(W, dW, lr=1e-2):
    W -= lr * dW
    return W

