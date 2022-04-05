import numpy as np
from tensorflow import keras
from keras.applications.vgg16 import VGG16
import pickle

# from keras.applications.vgg16 import VGG16

# (x_train, y_train) = keras.preprocessing.image_dataset_from_directory('D:/Thesis_data/training_img', batch_size=64)
x_infile = open("D:/Thesis_data/x_train.plk", 'rb')
y_infile = open("D:/Thesis_data/y_train.plk", 'rb')
x_train = pickle.load(x_infile)
y_train = pickle.load(y_infile)

print(np.shape(x_train))
print(np.shape(y_train))

model = VGG16(classes=14)

k = model.predict(x_train[0])
# model.summary()

