import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pathlib


img_height = 224
img_width = 224
input_shape = (img_width, img_height, 3)
batch_size = 16
train_dir = pathlib.Path('D:/Thesis_data/compartment_training')
test_dir  = pathlib.Path('D:/Thesis_data/compartment_testing')
checkpoint_dir = pathlib.Path('D:./TF_checkpoint/compartmentized/weight/')
model_dir = pathlib.Path('D:./TF_backup/compartment/compartment.h5')
model_plot_dir = pathlib.Path('D:./TF_backup/compartment/compartment.png')
classes = ['Compartmentized', 'Plated']

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    seed=665,
    shuffle=True,
    validation_split=0.25,
    subset="training",
    image_size=(img_height, img_width),
    batch_size=batch_size
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)


model_checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_dir,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
)

# for img,lab in train_ds.take(1):
#     print(np.shape(img))


Gx = np.array([[
    [-1, 0, 1,-2, 0, 2,-1, 0, -1]
]], dtype=np.float32)
Gy = np.array([[
    [1, 2, 1, 0, 0, 0, -1, -2, -1]
]], dtype=np.float32)
Gx = np.multiply(Gx, 0.125)
Gx = tf.Variable(tf.constant(Gx, shape=[3, 3, 1, 1]))
Gy = np.multiply(Gy, 0.125)
Gy = tf.Variable(tf.constant(Gy, shape=[3, 3, 1, 1]))

#input
input_layer_rgb = layers.Input(shape=input_shape)
input_layer = layers.Rescaling(scale=1./255, offset=0)(input_layer_rgb)
#block 1
grey = tf.image.rgb_to_grayscale(input_layer_rgb)
contrast = tf.image.adjust_contrast(grey, 1.5)
# satu = tf.image.adjust_saturation(input_layer, 1.5)
# bri = tf.image.adjust_brightness(input_layer,-30)
gx = tf.nn.conv2d(contrast,Gx, strides=[1, 1, 1, 1], padding='VALID')
thresX = layers.ThresholdedReLU(theta=5)(gx)
gy = tf.nn.conv2d(contrast,Gy, strides=[1, 1, 1, 1], padding='VALID')
thresY = layers.ThresholdedReLU(theta=5)(gy)
add = layers.Add()([thresX,thresY])
conv = layers.Conv2D(56, kernel_size=50, activation="relu")(add)
pool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv)
conv = layers.Conv2D(32, kernel_size=25, activation="relu")(pool)
pool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv)
conv = layers.Conv2D(14, kernel_size=15, activation="relu")(pool)
pool = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv)
flat1 = layers.Flatten()(pool)

#block2
# hsv = tf.image.rgb_to_hsv(input_layer_rgb)
# hsv = layers.Rescaling(scale=1./255, offset=0)(hsv)
# conv2 = layers.Conv2D(56, kernel_size=50, activation="relu")(hsv)
# pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
# conv2 = layers.Conv2D(32, kernel_size=25, activation="relu")(pool2)
# pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
# conv2 = layers.Conv2D(14, kernel_size=15, activation="relu")(pool2)
# pool2 = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)
# flat2 = layers.Flatten()(pool2)

# conc = layers.concatenate([flat1, flat2], axis=1)
dense = layers.Dense(946, activation="relu")(flat1)
dense = layers.Dense(512, activation="relu")(dense)
dense = layers.Dense(64, activation="relu")(dense)
dense = layers.Dense(2, activation="softmax")(dense)


model = keras.Model(input_layer_rgb, dense)
model.summary()
tf.keras.utils.plot_model(model, to_file=model_plot_dir, show_shapes=True)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


epochs = 8
model.fit(train_ds, epochs=epochs, callbacks=[model_checkpoint])
model.save(model_dir)

test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

score = model.evaluate(test_ds, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])