from NeuralNetwork import NeuralNetwork
import numpy as np

if __name__ == '__main__':
    train = np.array([[-2, -1],  #Alice
                    [25, 6],   #Bob
                    [17, 4],   #Charlie
                    [-15, -6],]) #Diana
    expected = np.array([1,    #Alice
                         0,    #Bob
                         0,    #Chalie
                         1])   #Diana

    network = NeuralNetwork()
    network.train(train, expected)

    ana = np.array([-7,-3])
    joan = np.array([20, 3])

    print("Ana: %.3f" % network.feedforward(ana))
    print("Joan: %.3f" % network.feedforward(joan))
