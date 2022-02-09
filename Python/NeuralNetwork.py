import numpy as np
from numpy import random as rd

def sigmoid(x):
    return 1/(1+np.power(np.e, -x))

class MultilayerNeuralNetwork:
    def __init__(self, layer_array, learning_rate):
        self.layer_array = layer_array


        self.weight_matrix = []
        for i in range(0, len(layer_array)-1):
            self.weight_matrix.append(Matrix(layer_array[i+1], layer_array[i]).randomize())

        self.bias_matrix = []
        for i in range(0, len(layer_array)-1):
            self.bias_matrix.append(Matrix(layer_array[i+1], 1).randomize())
        self.learning_rate = learning_rate

    def feed_forward(self, input_array):
        if (len(input_array) != self.layer_array[0]):
            print("Wrong input dimension!")
            return -1
        else:
            data_matrix = Matrix.array_2_matrix(input_array);
            for i in range(len(self.weight_matrix)):
                data_matrix = np.dot(self.weight_matrix[i],data_matrix)
                data_matrix = np.add(data_matrix, np.array(self.bias_matrix[i].T).flatten())
                data_matrix = map(sigmoid, data_matrix)
            return data_matrix
    
    def train(self, input_array, target_array):
        if (len(input_array) != self.layer_array[0]):
            print("Wrong input dimension!")
            return -1
        elif (len(target_array) != self.layer_array[len(self.layer_array)-1]):
            print("Wrong target dimension!")
            return -2
        else:
            #feed-forward
            layer_result_matrix_array = np.array([])
            data_matrix = np.array(input_array)
            np.append(layer_result_matrix_array, data_matrix, axis=0)
            for i in range(len(self.weight_matrix)):
                data_matrix = np.dot(self.weight_matrix[i], data_matrix)
                data_matrix = np.add(data_matrix, np.array(self.bias_matrix[i].T).flatten())
                data_matrix = map(sigmoid, data_matrix)
                np.append(layer_result_matrix_array, data_matrix, axis=0)
            feed_result_matrix = data_matrix
            print(layer_result_matrix_array)


        




nn = MultilayerNeuralNetwork([5,4,3,2], 0.1)
# res = nn.feed_forward(np.transpose([1,2,3,4,5]))
nn.train(np.transpose([1,2,3,4,5]), [2, 3])

# print(res)

