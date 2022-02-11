
from neural_network import MultilayerNeuralNetwork, train_data


dataset = []
dataset.append(train_data([1,0], [1]))
dataset.append(train_data([0,1], [1]))
dataset.append(train_data([0,0], [0]))
dataset.append(train_data([1,1], [0]))

# mlnn = MultilayerNeuralNetwork([2, 4, 1], 1) 
# mlnn.batch_training(dataset, 8000, 2, 0.9995)

# inp = input("Save NN? Y/N: ")
# if (inp == "Y"):
#     mlnn.save_weight("xor_final.plk")


mlnn = MultilayerNeuralNetwork.load_weight("xor_final.plk")

print(mlnn.feed_forward(dataset[0].input_array))
print(mlnn.feed_forward(dataset[1].input_array))
print(mlnn.feed_forward(dataset[2].input_array))
print(mlnn.feed_forward(dataset[3].input_array))



