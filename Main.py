from NeuralNetwork import NeuralNetwork
import numpy as np

if __name__ == '__main__':
    network = NeuralNetwork()
    inputs = np.array([2,3])

    print(network.activate(inputs))
