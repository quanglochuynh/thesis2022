import numpy as np
from numpy import random as rd
from matrix import Matrix

def sigmoid(x):
    return 1/(1+np.power(np.e, -x))

def dsigmoid(x):
    return x*(1-x)

class MultilayerNeuralNetwork:
    def __init__(self, layer_array, learning_rate):
        self.layer_array = layer_array

        self.weight_matrix = []
        for i in range(0, len(layer_array)-1):
            m = Matrix(layer_array[i+1], layer_array[i])
            m.randomize()
            self.weight_matrix.append(m)
        
        self.bias_matrix = []
        for i in range(0, len(layer_array)-1):
            m = Matrix(layer_array[i+1], 1)
            m.randomize()
            self.bias_matrix.append(m)
            
        self.learning_rate = learning_rate



    def feed_forward(self, input_array):
        if (len(input_array) != self.layer_array[0]):
            print("Wrong input dimension!")
            return -1
        else:
            data_matrix = Matrix.array_2_matrix(input_array)
            # print(data_matrix.data)
            for i in range(len(self.weight_matrix)):
                data_matrix = Matrix.multiply(self.weight_matrix[i], data_matrix)
                data_matrix = Matrix.add(data_matrix, self.bias_matrix[i])
                data_matrix.data = map(sigmoid, data_matrix.data)
            return Matrix.matrix_2_array(data_matrix)
    
    def train(self, input_array, target_array):
        if (len(input_array) != self.layer_array[0]):
            print("Wrong input dimension!")
            return -1
        elif (len(target_array) != self.layer_array[len(self.layer_array)-1]):
            print("Wrong target dimension!")
            return -2
        else:
            #feed-forward
            layer_result_matrix_array = []

            feed_matrix = Matrix.array_2_matrix(input_array)
            layer_result_matrix_array.append(feed_matrix)
            for i in range(len(self.weight_matrix)):
                feed_matrix = Matrix.multiply(self.weight_matrix[i], feed_matrix)
                feed_matrix = Matrix.add(feed_matrix, self.bias_matrix[i])
                feed_matrix.data = map(sigmoid, feed_matrix.data)
                layer_result_matrix_array.append(feed_matrix)
            feed_result_matrix = feed_matrix
            #BACK-propagation
            target_matrix = Matrix.array_2_matrix(target_array)
            error_matrix = Matrix.subtract(target_matrix, feed_result_matrix)
            # print(error_matrix.data)
            for i in range(len(self.layer_array)-2, 0):
                gradient_matrix = Matrix.map(layer_result_matrix_array[i+1], dsigmoid)
                gradient_matrix.data = gradient_matrix.data * self.learning_rate
                delta = Matrix.multiply(gradient_matrix, Matrix.transpose(layer_result_matrix_array[i]))
                self.weight_matrix[i] = Matrix.add(self.weight_matrix[i], delta)
                self.bias_matrix[i] = Matrix.add(self.bias_matrix[i], gradient_matrix)
                if (i==-1): continue
                weight_transposed = Matrix.transpose(self.weight_matrix[i])
                error_matrix = Matrix.multiply(weight_transposed, error_matrix)
            return 0






nn = MultilayerNeuralNetwork([5,4,3,2], 0.1)
# res = nn.feed_forward([1,2,3,4,5])
nn.train(np.transpose([1,2,3,4,5]), [2, 3])



