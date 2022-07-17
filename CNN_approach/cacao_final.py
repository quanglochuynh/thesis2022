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
train_dir = pathlib.Path('D:/Thesis_data/others_dataset/training_img')
test_dir  = pathlib.Path('D:/Thesis_data/others_dataset/testing_img')
checkpoint_dir = pathlib.Path('D:./TF_checkpoint/finale/weight/')
model_plot_dir = pathlib.Path('D:./TF_backup/finale/final_model.png')
final_model_dir = pathlib.Path('D:./TF_backup/finale/final_model.h5')


# train_ds = keras.utils.image_dataset_from_directory(
#     train_dir,
#     labels="inferred",
#     label_mode="categorical",
#     seed=665,
#     shuffle=True,
#     validation_split=0.25,
#     subset="training",
#     image_size=(img_height, img_width),
#     batch_size=batch_size
# )
# train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# model_checkpoint = keras.callbacks.ModelCheckpoint(
#     checkpoint_dir,
#     save_weights_only=True,
#     monitor='val_accuracy',
#     mode='max',
# )

# Create model
color_model = keras.models.load_model(pathlib.Path('D:./TF_backup/finale/model6.h5'))
structure_model = keras.models.load_model(pathlib.Path('D:./TF_backup/finale/cpm.h5'))
others_model = keras.models.load_model(pathlib.Path('D:./TF_backup/finale/other.h5'))
decision_model = keras.models.load_model(pathlib.Path('D:./TF_backup/finale/dt.h5'))
decision_model._name = "decision_tree"

input_layer = keras.Input(shape=input_shape)
x = color_model(input_layer)
x._name = "color_model"
y = structure_model(input_layer)
y._name = "structure_model"
z = others_model(input_layer)
z._name = "others_model"
conc = tf.concat([x,y,z], axis=1)
dt = decision_model(conc)
final_model = keras.Model(input_layer,dt)

# final_model.summary()
tf.keras.utils.plot_model(y, to_file=model_plot_dir, show_shapes=True)

# final_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

