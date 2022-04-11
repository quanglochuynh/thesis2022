import pathlib
import tensorflow as tf
from tensorflow import keras
from keras import layers
# import numpy

img_height = 224
img_width = 224
input_shape = (img_width, img_height, 3)
batch_size = 8
train_dir = pathlib.Path('D:/Thesis_data/training_img')
test_dir  = pathlib.Path('D:/Thesis_data/testing_img')
checkpoint_dir = pathlib.Path('D:./TF_checkpoint/cacao_CNN/weight/')
model_dir = pathlib.Path('D:./TF_backup/cacao_CNN/')

train_ds = keras.utils.image_dataset_from_directory(
  train_dir,
  label_mode="categorical",
  shuffle=True,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# for img,lab in train_ds.take(1):
#     print(numpy.max(img[1]))

model_checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_dir,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_freq=700
)


input_layer = layers.Input(shape=input_shape)
input_layer = layers.Rescaling(scale=1./255, offset=0)(input_layer)
conv1 = layers.Conv2D(64, kernel_size=(6,6), activation="relu")(input_layer)
pool1 = layers.MaxPooling2D(pool_size=(4,4), strides=4)(conv1)
conv2 = layers.Conv2D(32, kernel_size=(24,24), activation="relu")(pool1)
pool2 = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv2)
flat = layers.Flatten()(pool2)
dense1 = layers.Dense(128, activation="relu")(flat)
dense2 = layers.Dense(64, activation="relu")(dense1)
dense3 = layers.Dense(32, activation="relu")(dense2)
output_layer = layers.Dense(14, activation="softmax")(dense3)

model = keras.Model(input_layer, output_layer)
# model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.load_weights(checkpoint_dir)

epochs = 1
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

# model = keras.models.load_model(model_dir)
score = model.evaluate(test_ds, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
