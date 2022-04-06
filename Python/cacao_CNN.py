import pathlib
from pickletools import optimize
import tensorflow as tf
from tensorflow import keras
from keras import layers

img_height = 224
img_width = 224
input_shape = (img_width, img_height, 3)
batch_size = 8
train_dir = pathlib.Path('D:/Thesis_data/training_img')
test_dir  = pathlib.Path('D:/Thesis_data/testing_img')

train_ds = keras.utils.image_dataset_from_directory(
  train_dir,
  label_mode="categorical",
  shuffle=True,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# test_ds = keras.utils.image_dataset_from_directory(
#     test_dir,
#     label_mode="categorical",
#     shuffle=True,
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )
# test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

input_layer = layers.Input(shape=input_shape)
input_layer = layers.Rescaling(1/255, offset=0)(input_layer)
conv1 = layers.Conv2D(64, kernel_size=(6,6), activation="relu")(input_layer)
pool1 = layers.MaxPooling2D(pool_size=(4,4), strides=4)(conv1)
conv2 = layers.Conv2D(32, kernel_size=(24,24), activation="relu")(pool1)
pool2 = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv2)
flat = layers.Flatten()(pool2)
dense1 = layers.Dense(28, activation="sigmoid")(flat)
dense2 = layers.Dense(28, activation="sigmoid")(dense1)
output_layer = layers.Dense(14, activation="softmax")(dense2)

model = keras.Model(input_layer, output_layer)
# model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 2
model.fit(train_ds, epochs = epochs)

model.save('D:/Tensorflow_backups')