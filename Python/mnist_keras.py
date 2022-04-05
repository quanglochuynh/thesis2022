import os.path
import numpy as np
import struct
import gzip as gz
from tensorflow import keras
from keras import layers

dr = "D:/Thesis_data/Backups/MNIST/"
# dir = "/Users/lochuynhquang/Documents/MNIST/"
weight_path = "weight/mnist_keras.ckpt"
model_dir = os.path.dirname(weight_path)
train_img_dir = "train-images-idx3-ubyte.gz"
train_lab_dir = "train-labels-idx1-ubyte.gz"
test_img_dir = "t10k-images-idx3-ubyte.gz"
test_lab_dir = "t10k-labels-idx1-ubyte.gz"

inp_training = gz.open(dr + train_img_dir, 'r')

input_shape = (28, 28, 1)
num_classes = 10


def read_idx(filename):
    with gz.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


# data preparation
x_train = read_idx(dr + train_img_dir)
y_train = read_idx(dr + train_lab_dir)
x_test = read_idx(dr + test_img_dir)
y_test = read_idx(dr + test_lab_dir)

x_train = x_train.astype("float32") / 255
x_test_data = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# # define model
# model = keras.Sequential(
#     [
#         keras.Input(shape=input_shape),
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )
#
# # Training
# batch_size = 128
# epochs = 1
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# model = keras.models.load_model("weight/mnist_keras.tf")

# # Testing
# score = model.evaluate(x_test, y_test, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])

print(type(x_train[4][5][4][0]))

# model.save('weight/mnist_keras.tf')
