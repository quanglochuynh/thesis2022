from neural_network import MultilayerNeuralNetwork, train_data
import numpy as np

os = "macos"

name = ["C/Am", "C#/A#m", "D/Bm", "D#/Cm", "E/C#m", "F/Dm", "F#/D#m", "G/Em", "G#/Fm", "A/F#m", "A#/Gm", "B/G#m"]


# data = [1,0,1,0,1,1,0,1,0,1,0,1]
# dataset = []

# for i in range(12):
#     target = [0]*12
#     target[i] = 1
#     dataset.append(train_data(data, target))
#     print(dataset[i].input_array)
#     data.insert(0,data.pop())

# print("\n")

# for i in range(12):
#     print(dataset[i].input_array)

# dataset.append(train_data([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1,0,0,0,0,0,0,0,0,0,0,0]))
# dataset.append(train_data([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0], [0,1,0,0,0,0,0,0,0,0,0,0]))
# dataset.append(train_data([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1], [0,0,1,0,0,0,0,0,0,0,0,0]))
# dataset.append(train_data([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0,0,0,1,0,0,0,0,0,0,0,0]))
# dataset.append(train_data([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1], [0,0,0,0,1,0,0,0,0,0,0,0]))
# dataset.append(train_data([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0], [0,0,0,0,0,1,0,0,0,0,0,0]))
# dataset.append(train_data([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1], [0,0,0,0,0,0,1,0,0,0,0,0]))
# dataset.append(train_data([1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1], [0,0,0,0,0,0,0,1,0,0,0,0]))
# dataset.append(train_data([1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0,0,0,0,0,0,0,0,1,0,0,0]))
# dataset.append(train_data([0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1], [0,0,0,0,0,0,0,0,0,1,0,0]))
# dataset.append(train_data([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], [0,0,0,0,0,0,0,0,0,0,1,0]))
# dataset.append(train_data([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1], [0,0,0,0,0,0,0,0,0,0,0,1]))

# mlnn = MultilayerNeuralNetwork([12, 12, 12], 0.1)
mlnn = MultilayerNeuralNetwork.load_weight("scale_identifier_final.plk", os)
# mlnn.batch_training(dataset, 24000, 1, 0.9998)

# inp = input("Save NN? Y/N: ")
# if (inp == "Y"):
#     mlnn.save_weight("scale_identifier_final.plk")



n = ""
inp = [0]*12
while (n!="0"):
    n = input("Type the note: ")
    if (n=="c"):
        inp[0] = 1
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))])
    elif (n == "c#"):
        inp[1] = 1
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))])
    elif (n == "d"):
        inp[2] = 1
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))])
    elif (n == "d#"):
        inp[3] = 1
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))])
    elif (n == "e"):
        inp[4] = 1
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))])
    elif (n == "f"):
        inp[5] = 1
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))])
    elif (n == "f#"):
        inp[6] = 1
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))])
    elif (n == "g"):
        inp[7] = 1
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))])
    elif (n == "g#"):
        inp[8] = 1   
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))]) 
    elif (n == "a"):
        inp[9] = 1
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))])
    elif (n == "a#"):
        inp[10] = 1
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))])
    elif (n == "b"):
        inp[11] = 1
        res = mlnn.feed_forward(inp).flatten().tolist()
        print(name[res.index(max(res))])
    else:
        exit()

    