import numpy as np
from numpy import random as rd
import cupy as cp
import pickle
from gpu_matrix import GPUMatrix

def sigmoid(x):
    return 1/(1+cp.power(cp.e, -x))

def dsigmoid(x):
    return x*(1-x)

class GPUMultilayerNeuralNetwork:
    def __init__(self, layer_array, learning_rate):
        self.layer_array = layer_array

        self.weight_matrix = []
        for i in range(0, len(layer_array)-1):
            m = GPUMatrix(layer_array[i+1], layer_array[i])
            m.randomize()
            self.weight_matrix.append(m)
        
        self.bias_matrix = []
        for i in range(0, len(layer_array)-1):
            m = GPUMatrix(layer_array[i+1], 1)
            m.randomize()
            self.bias_matrix.append(m)
            
        self.learning_rate = learning_rate

    def feed_forward(self, input_array):
        if (len(input_array) != self.layer_array[0]):
            print("Wrong input dimension!")
            return -1
        else:
            data_matrix = GPUMatrix.array_2_matrix(input_array)
            # print(data_matrix.data)
            for i in range(len(self.weight_matrix)):
                data_matrix = GPUMatrix.multiply(self.weight_matrix[i], data_matrix)
                data_matrix = GPUMatrix.add(data_matrix, self.bias_matrix[i])
                data_matrix = GPUMatrix.map(data_matrix, sigmoid)
            return GPUMatrix.matrix_2_array(data_matrix)
    
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
            feed_matrix = GPUMatrix.array_2_matrix(input_array)
            layer_result_matrix_array.append(feed_matrix)
            for i in range(len(self.weight_matrix)):
                feed_matrix = GPUMatrix.multiply(self.weight_matrix[i], feed_matrix)
                feed_matrix = GPUMatrix.add(feed_matrix, self.bias_matrix[i])
                feed_matrix = GPUMatrix.map(feed_matrix, sigmoid)
                layer_result_matrix_array.append(feed_matrix)
            feed_result_matrix = layer_result_matrix_array[len(layer_result_matrix_array)-1]
            #BACK-propagation
            target_matrix = GPUMatrix.array_2_matrix(target_array)
            error_matrix = GPUMatrix.subtract(target_matrix, feed_result_matrix)
            for i in range(len(self.layer_array)-2, -1, -1):
                gradient_matrix = GPUMatrix.map(layer_result_matrix_array[i+1], dsigmoid)
                gradient_matrix = GPUMatrix.hadamard(gradient_matrix, error_matrix)
                gradient_matrix.data = cp.multiply(gradient_matrix.data, self.learning_rate)
                delta = GPUMatrix.multiply(gradient_matrix, GPUMatrix.transpose(layer_result_matrix_array[i]))
                self.weight_matrix[i] = GPUMatrix.add(self.weight_matrix[i], delta)
                self.bias_matrix[i] = GPUMatrix.add(self.bias_matrix[i], gradient_matrix)
                weight_transposed = GPUMatrix.transpose(self.weight_matrix[i])
                error_matrix = GPUMatrix.multiply(weight_transposed, error_matrix)
            return 0

    def batch_training(self, test_array, n, initial_lr, damping_coeficient):
        print("Training...")
        original_lr = self.learning_rate
        lr = initial_lr
        self.learning_rate = lr
        for i in range(n):
            data = rd.choice(test_array)
            self.train(data.input_array, data.target_array)
            self.learning_rate = self.learning_rate * damping_coeficient
        self.learning_rate = original_lr
        #accuracy testing
        deviation = cp.zeros(cp.shape(test_array[0].target_array))
        print("Result: ")
        for i in range(len(test_array)):
            res = self.feed_forward(test_array[i].input_array)
            print(res.flatten())
            error = cp.subtract(test_array[i].target_array, res)
            deviation =cp.add(deviation, cp.absolute(error))
        print("\nNet error: " +  str(deviation.flatten().flatten()))
    
    def save_weight(self, name):
        filename = "weight/"+name
        outfile = open(filename, 'wb')
        pickle.dump(self, outfile)

    @staticmethod
    def load_weight(filename, os):      #true for macOS, false for Windows
        if (os == "macos"):
            infile = open("/Users/lochuynhquang/Documents/thesis2022/Python/weight/" + filename, 'rb')
        else:
            infile = open("/Users/quang/Documents/thesis2022/Python/weight/" + filename, 'rb')
        return pickle.load(infile)




class train_data:
    def __init__(self, input_array, target_array):
        self.input_array = input_array
        self.target_array = target_array


# mlnn = GPUMultilayerNeuralNetwork([2,2,2], 0.1)

# print(mlnn.feed_forward([1,0]))
