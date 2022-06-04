import numpy as np
from numpy import random as rd
import os
import pickle
from matrix import Matrix
from sklearn.utils import shuffle

def sigmoid(x):
    return 1/(1+pow(2.7182818284, -x))

def dsigmoid(x):
    return x*(1-x)

def relu(x):
    return max(x,0)

def drelu(x):
    if x<=0:
        return 0
    else:
        return 1

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

    def predict(self, input_array):
        if (len(input_array) != self.layer_array[0]):
            print("Wrong input dimension!")
            return -1
        else:
            data_matrix = Matrix.array_2_matrix(input_array)
            # print(data_matrix.data)
            for i in range(len(self.weight_matrix)):
                data_matrix = Matrix.multiply(self.weight_matrix[i], data_matrix)
                data_matrix = Matrix.add(data_matrix, self.bias_matrix[i])
                data_matrix = Matrix.map(data_matrix, sigmoid)
            return Matrix.matrix_2_array(data_matrix)
    
    def batch_train(self, input_array, target_array, activation=sigmoid, dactivation=dsigmoid):
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
                feed_matrix = Matrix.map(feed_matrix, activation)
                layer_result_matrix_array.append(feed_matrix)
            feed_result_matrix = layer_result_matrix_array[len(layer_result_matrix_array)-1]
            #BACK-propagation
            target_matrix = Matrix.array_2_matrix(target_array)
            error_matrix = Matrix.subtract(target_matrix, feed_result_matrix)
            for i in range(len(self.layer_array)-2, -1, -1):
                gradient_matrix = Matrix.map(layer_result_matrix_array[i+1], dactivation)
                gradient_matrix = Matrix.hadamard(gradient_matrix, error_matrix)
                gradient_matrix.data = np.multiply(gradient_matrix.data, self.learning_rate)
                delta = Matrix.multiply(gradient_matrix, Matrix.transpose(layer_result_matrix_array[i]))
                self.weight_matrix[i] = Matrix.add(self.weight_matrix[i], delta)
                self.bias_matrix[i] = Matrix.add(self.bias_matrix[i], gradient_matrix)
                weight_transposed = Matrix.transpose(self.weight_matrix[i])
                error_matrix = Matrix.multiply(weight_transposed, error_matrix)
            return np.sum(np.square(error_matrix.data))

    def fit(self,dataset, epochs=1, initial_lr=0.01, damping_coeficient=1, activation=sigmoid, dactivation=dsigmoid):
        print("Training...")
        original_lr = self.learning_rate
        lr = initial_lr
        self.learning_rate = lr
        x,y = dataset.get()
        for i in range(epochs):
            print('Epoch ' + str(i+1) + '/' + str(epochs))
            for j in range(dataset.length):
                err = self.batch_train(x,y)
                self.learning_rate = self.learning_rate * damping_coeficient
                print('Training accuracy:', err)
        self.learning_rate = original_lr
        #accuracy testing
        # deviation = np.zeros(np.shape(test_array[0].target_array))
        # print("Result: ")
        # for i in range(100):
        #     data = rd.choice(test_array)
        #     res = self.feed_forward(test_array[i].input_array)
        #     # print(res.flatten())
        #     error = np.subtract(test_array[i].target_array, res)
        #     deviation =np.add(deviation, np.absolute(error))
        # print("\nNet error: " +  str(deviation.flatten().flatten()))
    

    def save_weight(self, name):
        filename = "weight/"+name
        outfile = open(filename, 'wb')
        pickle.dump(self, outfile)

    @staticmethod
    def load_weight(filename, os):      #true for macOS, false for Windows
        cwd = os.getcwd()
        infile = open(cwd + "/weight/" + filename, 'rb')
        return pickle.load(infile)

class dataset:
    def __init__(self, x, y):
        self.length = np.shape(x)[0]
        self.id = 0
        self.x, self.y = shuffle(x,y,random_state=0)
        
    def get(self):
        self.id = self.id+1
        if (self.id>self.length): id=0
        return self.x[self.id-1], self.y[self.id-1]