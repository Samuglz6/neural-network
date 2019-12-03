from NeuralNetwork import NeuralNetwork
import numpy as np

if __name__ == '__main__':
    train = np.array([[-8, -1],  #Alice
                    [19, 6],   #Bob
                    [11, 4],   #Charlie
                    [-21, -6],]) #Diana
    expected = np.array([[1,    #Alice
                         0,    #Bob
                         0,    #Chalie
                         1],])   #Diana

    network = NeuralNetwork()
    network.train(train, expected)

    input = np.array([-8,-1])
    #print(network.feedforward(input))
