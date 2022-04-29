import pathlib
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

img_height = 224
img_width = 224
input_shape = (img_width, img_height, 3)
batch_size = 16
train_dir = pathlib.Path('D:/Thesis_data/compartment_training')
test_dir  = pathlib.Path('D:/Thesis_data/compartment_testing')
checkpoint_dir = pathlib.Path('D:./TF_checkpoint/compartmentized/weight/')
model_dir = pathlib.Path('D:./TF_backup/compartment/compartment.h5')
model_plot_dir = pathlib.Path('D:./TF_backup/compartment/compartment.png')


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

# # for img,lab in train_ds.take(1):
# #     print(numpy.max(img[1]))

model_checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_dir,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
)

# Create model
rgb_input = layers.Input(shape=input_shape)

# hsv_input = tf.image.rgb_to_hsv(rgb_input)
# rescale_rgb = layers.Rescaling(scale=1./255, offset=0)(rgb_input)
# rescale_hsv = layers.Rescaling(scale=1./255, offset=0)(hsv_input)

# conv1rgb = layers.Conv2D(28, kernel_size=(7,7), activation="relu")(rescale_rgb)
# pool1rgb = layers.MaxPooling2D(pool_size=(4,4), strides=2)(conv1rgb)
# conv2rgb = layers.Conv2D(14, kernel_size=(5,5), activation="relu")(pool1rgb)
# pool2rgb = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv2rgb)
# conv3rgb = layers.Conv2D(7, kernel_size=(3,3), activation="relu")(pool2rgb)
# pool3rgb = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv3rgb)

# conv1hsv = layers.Conv2D(14, kernel_size=(7,7), activation="relu")(rescale_hsv)
# pool1hsv = layers.MaxPooling2D(pool_size=(4,4), strides=2)(conv1hsv)
# conv2hsv = layers.Conv2D(7, kernel_size=(5,5), activation="relu")(pool1hsv)
# pool2hsv = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv2hsv)
# conv3hsv = layers.Conv2D(3, kernel_size=(3,3), activation="relu")(pool2hsv)
# pool3hsv = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv3hsv)

# concat = layers.concatenate([pool3rgb, pool3hsv], axis=1)
# flat = layers.Flatten()(concat)
# drop = layers.Dropout(0.5)(flat)
# dense3 = layers.Dense(96, activation="relu")(drop)
# drop2 = layers.Dropout(0.4)(dense3)
# dense4 = layers.Dense(48, activation="relu")(drop2)
# drop3 = layers.Dropout(0.2)(dense4)
# output_layer = layers.Dense(6, activation="softmax")(drop3)

model = keras.Model(rgb_input, output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()

# tf.keras.utils.plot_model(model, to_file=model_plot_dir, show_shapes=True)
# Load model
# model.load_weights(checkpoint_dir)
# model = keras.models.load_model(model_dir)
# model.summary()

## Train model
# epochs = 10
# model.fit(train_ds, epochs=epochs, callbacks=[model_checkpoint])
# model.save(model_dir)


# ## Test model
# test_ds = keras.utils.image_dataset_from_directory(
#     test_dir,
#     label_mode="categorical",
#     shuffle=True,
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )
# test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# score = model.evaluate(test_ds, verbose=1)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])