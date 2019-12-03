from Neuron import Neuron
import numpy as np

class NeuralNetwork:
    def __init__(self):
        weights = np.array([0,1])
        bias = 0

        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def activate(self, inputs):
        out_h1 = self.h1.activate(inputs)
        out_h2 = self.h2.activate(inputs)

        out_o1 = self.o1.activate(np.array([out_h1, out_h2]))

        return out_o1
