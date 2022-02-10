import numpy as np
from numpy import random as rd

from neural_network import MultilayerNeuralNetwork


class Epoch:
    def __init__(self, input_array, target_array):
        self.input_array = input_array
        self.target_array = target_array


# nn = MultilayerNeuralNetwork([5,4,3,2], 0.1)
# # res = nn.feed_forward([1,2,3,4,5])
# nn.train(np.transpose([1,2,3,4,5]), [2, 3])

dataset = []
dataset.append(Epoch([1,0], [1]))
dataset.append(Epoch([0,1], [1]))
dataset.append(Epoch([0,0], [0]))
dataset.append(Epoch([1,1], [0]))

mlnn = MultilayerNeuralNetwork([2, 2, 1], 0.1)

# print(mlnn.feed_forward(dataset[0].input_array))
# print(mlnn.feed_forward(dataset[1].input_array))
# print(mlnn.feed_forward(dataset[2].input_array))
# print(mlnn.feed_forward(dataset[3].input_array))

for i in range(1):
    data = rd.choice(dataset)
    # print(data.input_array)
    # print(data.target_array)
    mlnn.train(data.input_array, data.target_array)

# print(mlnn.feed_forward(dataset[0].input_array))
# print(mlnn.feed_forward(dataset[1].input_array))
# print(mlnn.feed_forward(dataset[2].input_array))
# print(mlnn.feed_forward(dataset[3].input_array))