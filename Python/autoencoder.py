from asyncore import close_all
import numpy as np
import cv2
from neural_network import MultilayerNeuralNetwork, train_data

# image = cv2.imread('/Users/lochuynhquang/Documents/plantvillage/Corn___Common_rust/image (1).JPG')
image = cv2.imread('C:\PlantVillage\Corn___healthy\image (1).JPG')
image = cv2.resize(image, (128,128), interpolation=cv2.INTER_LINEAR)
cv2.imshow('original', image)


print(image)
height, width, depth = np.shape(image)
inp = np.reshape(image, (height*width*depth))
# print(inp)


out = inp/2
# print(out)

out = np.reshape(out, (128,128,3))
# print(out)

cv2.imshow('test', np.uint8(out))

cv2.waitKey(0)
cv2.destroyAllWindows()