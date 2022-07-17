import pathlib
import tensorflow as tf
from tensorflow import keras
from keras import layers
# import cv2
# import numpy as np

img_height = 224
img_width = 224
input_shape = (img_width, img_height, 3)
batch_size = 8
train_dir = pathlib.Path('D:/Thesis_data/training_img')
test_dir  = pathlib.Path('D:/Thesis_data/testing_img')
checkpoint_dir = pathlib.Path('D:./TF_checkpoint/cacao_CAE/weight/')
model_dir = pathlib.Path('D:./TF_backup/cacao_CAE/')

train_ds = keras.utils.image_dataset_from_directory(
  train_dir,
  labels=None,
  shuffle=True,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

model_checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_dir,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_freq=700
)


input_layer = layers.Input(shape=input_shape)
input_layer = layers.Rescaling(scale=1./255, offset=0)(input_layer)
# encoder
x = layers.Conv2D(112, kernel_size=(3,3), activation="relu", padding="same")(input_layer)

x = layers.Conv2D(96, kernel_size=(5,5), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

x = layers.Conv2D(28, kernel_size=(5,5), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

x = layers.Conv2D(14, kernel_size=(3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

x = layers.Flatten()(x)
x = layers.Dense(56, activation="relu")(x)
x = layers.Dense(28, activation="relu")(x)
# #latent
latent = layers.Dense(14, activation="sigmoid")(x)
# #decoder
x = layers.Dense(28, activation="relu")(latent)
x = layers.Dense(56, activation="relu")(x)
x = layers.Dense(2352, activation="relu")(x)
x = layers.Reshape((14,14,12))(x)
x = layers.Conv2DTranspose(14, kernel_size=(3,3), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(28, kernel_size=(5,5), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(96, kernel_size=(5,5), strides=2, activation="relu", padding="same")(x)
x = layers.Conv2DTranspose(112, kernel_size=(3,3), strides=2, activation="relu", padding="same")(x)

output_layer = layers.Rescaling(scale=255, offset=0)(x)
output_layer = layers.Conv2D(3, kernel_size=(3,3), padding="same")(output_layer)

model = keras.Model(input_layer, output_layer)
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 1
i=0
j=0
for ep in range(epochs):
  j = j+1
  print("Epoch",j, "/", epochs)
  for batch in train_ds.take(5000):
      i = i+1
      score = model.train_on_batch(x=batch, y=batch, reset_metrics=False, return_dict=False)
      print("batch",i, "--Loss:",score[0])

model.save(model_dir)


# # model = keras.models.load_model(model_dir)

# for img in train_ds.take(1):
#     image = img
#     break

# result = model.predict(image, batch_size=None)

# # print(np.shape(result))
# image = np.asarray(image[0], dtype="uint8")
# cv2.imshow('original', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# result = np.asarray(result[0], dtype="uint8")
# cv2.imshow('result', result)
# print(np.max(result))
# print(np.min(result))
# cv2.waitKey(0)
# cv2.destroyAllWindows()