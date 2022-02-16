from asyncore import close_all
import numpy as np
import cv2
from neural_network import MultilayerNeuralNetwork, train_data

image = cv2.imread('/Users/lochuynhquang/Documents/plantvillage/Corn___Common_rust/image (1).JPG')
cv2.imshow('original', image)
cv2.waitKey(0)
# print(np.shape(image))
height, width, depth = np.shape(image)

im1d = np.reshape(image, -1)
# print(np.shape(im1d))

im1d = np.divide(im1d,255)

mlnn = MultilayerNeuralNetwork([196608, 196608], 0.1)

print('feeding')
out = mlnn.feed_forward(im1d)
out = np.multiply(out,255)

imout = np.reshape(out, (256,256,3))

cv2.imshow('fed', imout)

cv2.waitKey(0)
cv2.destroyAllWindows()