import pathlib
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications.vgg16 import VGG16


train_dir = pathlib.Path('D:/Thesis_data/training_img')
img_height = 224
img_width = 224
input_shape = (img_width, img_height,3)
batch_size = 8

train_ds = keras.utils.image_dataset_from_directory(
  train_dir,
  label_mode="categorical",
  shuffle=True,
  image_size=(img_height, img_width),
  batch_size=batch_size
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

base_model = VGG16(weights="imagenet", include_top=True)

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
