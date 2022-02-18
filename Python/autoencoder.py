from asyncore import close_all
import numpy as np
import cv2
from gpu_mlnn import GPUMultilayerNeuralNetwork, train_data
from neural_network import MultilayerNeuralNetwork


# image = cv2.imread('/Users/lochuynhquang/Documents/plantvillage/Corn___Common_rust/image (1).JPG')
image = cv2.imread('C:\PlantVillage\Corn___Northern_Leaf_Blight\image (3).JPG')
image = cv2.resize(image, (64,64), interpolation=cv2.INTER_LINEAR)
cv2.imshow('original', image)

#GPU
height, width, depth = np.shape(image)
inp = np.reshape(image, (height*width*depth))
inp = inp/255

mlnn = GPUMultilayerNeuralNetwork([12288, 8192, 2048, 128, 4, 128, 2048, 8192, 12288], 0.1)

out = mlnn.feed_forward(inp)
out = out*255
out = np.reshape(out, (64,64,3))
cv2.imshow('test', np.uint8(out))

#CPU
# height, width, depth = np.shape(image)
# inp = np.reshape(image, (height*width*depth))
# inp = inp/255

# mlnn = MultilayerNeuralNetwork([12288, 8192, 2048, 128, 4, 128, 2048, 8192, 12288], 0.1)

# out = mlnn.feed_forward(inp)
# out = out*255
# out = np.reshape(out, (64,64,3))
# cv2.imshow('test', np.uint8(out))


cv2.waitKey(0)
cv2.destroyAllWindows()