import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    return x * (1 - x)

def squad_error(obtained, expected):
    return (expected**2 - obtained**2)/2

class NeuralNetwork:
    def __init__(self):
        #Inicializamos los pesos para cada rama de forma aleatoria
        #Las 2 neuronas ocultas h1 y h2 tienen 2 entradas (2 pesos) y la
        #neurona de salida o1 otras 2 entradas.
        wh1 = [np.random.normal(), np.random.normal()]
        wh2 = [np.random.normal(), np.random.normal()]
        wo1 = [np.random.normal(), np.random.normal()]

        #Creamos las neuronas h1 y h2 de la capa oculta y o1 de la capa de salida
        self.h1 = Neuron(wh1, np.random.normal())
        self.h2 = Neuron(wh2, np.random.normal())
        self.o1 = Neuron(wo1, np.random.normal())

    def feedforward(self, x):
        sum_h1 = self.h1.weights[0] * x[0] + self.h1.weights[1] * x[1] + self.h1.bias
        f_h1 = self.h1.activationFunction(sum_h1)

        sum_h2 = self.h2.weights[0] * x[0] + self.h2.weights[1] * x[1] + self.h2.bias
        f_h2 = self.h2.activationFunction(sum_h2)

        sum_o1 = self.o1.weights[0] * x[0] + self.o1.weights[1] * x[1] + self.o1.bias
        result = self.o1.activationFunction(sum_o1)

        return result

    def train(self, input, expected):
        learn_rate = 0.5
        epochs = 1000

        for epoch in range(epochs):
            for x, y in zip(input, expected):
                 #FEEDFORWARD
                 sum_h1 = self.h1.weights[0] * x[0] + self.h1.weights[1] * x[1] + self.h1.bias
                 f_h1 = self.h1.activationFunction(sum_h1)

                 sum_h2 = self.h2.weights[0] * x[0] + self.h2.weights[1] * x[1] + self.h2.bias
                 f_h2 = self.h2.activationFunction(sum_h2)

                 sum_o1 = self.o1.weights[0] * x[0] + self.o1.weights[1] * x[1] + self.o1.bias
                 f_o1 = self.o1.activationFunction(sum_o1)

                 #BACKPROPAGATION
                 e = squad_error(f_o1, y)

                 #Variacion en los pesos de la Neurona o1
                 delta_o1 = e * deriv_sigmoid(f_o1)

                 d_o1_w1 = learn_rate * delta_o1 * f_h1
                 d_o1_w2 = learn_rate * delta_o1 * f_h2

                 new_weights_o1 = [(self.o1.weights[0] + d_o1_w1), (self.o1.weights[1] + d_o1_w2)]

                 #Variacion en los pesos de la Neurona h1
                 delta_h1 = deriv_sigmoid(f_h1) * (delta_o1 * self.o1.weights[0])

                 d_h1_w1 = learn_rate * delta_h1 * x[0]
                 d_h1_w2 = learn_rate * delta_h1 * x[1]

                 new_weights_h1 = [self.o1.weights[0] + d_o1_w1, self.o1.weights[1] + d_o1_w2]

                 #Variacion en los pesos de la Neurona h2
                 delta_h2 = deriv_sigmoid(f_h2) * (delta_o1 * self.o1.weights[1])

                 d_h2_w1 = learn_rate * delta_h2 * x[0]
                 d_h2_w2 = learn_rate * delta_h2 * x[1]

                 new_weights_h2 = [self.o1.weights[0] + d_o1_w1, self.o1.weights[1] + d_o1_w2]

                 #Actualizamos los pesos de las neuronas
                 self.o1.weights = new_weights_o1
                 self.h1.weights = new_weights_h1
                 self.h2.weights = new_weights_h2

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activationFunction(self, x):
        return sigmoid(x)
