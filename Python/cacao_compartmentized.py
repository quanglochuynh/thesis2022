import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pathlib

img_height = 224
img_width = 224
input_shape = (img_width, img_height, 3)
batch_size = 32
train_dir = pathlib.Path('D:/Thesis_data/structure_dataset/training_img')
test_dir  = pathlib.Path('D:/Thesis_data/structure_dataset/testing_img')
checkpoint_dir = pathlib.Path('D:/TF_checkpoint/compartmentized')
model_dir = pathlib.Path('D:/TF_backup/compartment/cpm2.h5')
model_plot_dir = pathlib.Path('D:/TF_backup/compartment/cpm.png')
classes = ["Compartmentize", "Plated", "Others"]


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

# # # for img,lab in train_ds.take(1):
# # #     print(numpy.max(img[1]))

# model_checkpoint = keras.callbacks.ModelCheckpoint(
#     checkpoint_dir,
#     save_weights_only=True,
#     monitor='val_accuracy',
#     mode='max',
# )

input_layer = layers.Input(shape=input_shape)
bri = tf.image.adjust_brightness(input_layer, 15)
hsv = tf.image.rgb_to_hsv(input_layer)

edge = tf.image.sobel_edges(bri)
thres = layers.ThresholdedReLU(theta=40)(edge)
split1, split2 = tf.split(value=thres, num_or_size_splits=[1,1], axis=4)
output1 = tf.squeeze(split1, axis=4)
output2 = tf.squeeze(split2, axis=4)

# Block1
conv1rgb = layers.Conv2D(25, kernel_size=(7,7), activation="relu")(output1)
pool1rgb = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv1rgb)
conv15rgb = layers.Conv2D(25, kernel_size=(7,7), activation="relu")(output2)
pool15rgb = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv15rgb)
add = layers.Add()([pool1rgb, pool15rgb])
conv2rgb = layers.Conv2D(50, kernel_size=(25,25), activation="relu")(add)
pool2rgb = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv2rgb)
conv3rgb = layers.Conv2D(28, kernel_size=(9,9), activation="relu")(pool2rgb)
pool3rgb = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv3rgb)

#Block2
conv1hsv = layers.Conv2D(25, kernel_size=(7,7), activation="relu")(hsv)
pool1hsv = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv1rgb)
conv2hsv = layers.Conv2D(50, kernel_size=(25,25), activation="relu")(pool1hsv)
pool2hsv = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv2hsv)
conv3hsv = layers.Conv2D(28, kernel_size=(9,9), activation="relu")(pool2hsv)
pool3hsv = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv3hsv)

# concat

conc = layers.Concatenate(axis=1)([pool3rgb, pool3hsv])

#Classification Block
flat = layers.Flatten()(conc)
drop = layers.Dropout(0.5)(flat)
dense3 = layers.Dense(512, activation="relu")(drop)
drop2 = layers.Dropout(0.4)(dense3)
dense4 = layers.Dense(64, activation="relu")(drop2)
drop3 = layers.Dropout(0.2)(dense4)
output_layer = layers.Dense(3, activation="softmax")(drop3)

model = keras.Model(input_layer, output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Load model
# model.load_weights(checkpoint_dir)
# model = keras.models.load_model(model_dir)
model.summary()


# # Train model
epochs = 1
model.fit(train_ds, epochs=epochs)
model.save(model_dir)
tf.keras.utils.plot_model(model, to_file=model_plot_dir, show_shapes=True)


## Test model
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

# # draw confusion matrix
# i = 0
# cmatrix = np.zeros((3,3), dtype=np.int16)
# for img,lab in test_ds.take(20):
#     result = model.predict(img, use_multiprocessing=True)
#     for j in range(np.size(result,axis=0)):
#         id1 = np.argmax(result[j])
#         id2 = np.argmax(lab[j])
#         # print(result[j], lab[j].numpy())
#         cmatrix[id1][id2] = cmatrix[id1][id2] + 1
# cmatrix = np.absolute(cmatrix)
# print(repr(cmatrix))

# fig = plt.figure(figsize=(6,6))
# plt.imshow(cmatrix)
# plt.title("Plot 2D array")
# plt.show()