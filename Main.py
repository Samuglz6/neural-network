from Neuron import Neuron
import numpy as np

if __name__ == '__main__':
    weights = np.array([1,2])
    bias = 4
    
    neuron = Neuron(weights, bias)

    inputs = np.array([1,1])
    print(neuron.activate(inputs))
    
