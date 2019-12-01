import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs):
        total = np.dot(self.weights, inputs) + self.bias
        return self.sigmoid(total)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
