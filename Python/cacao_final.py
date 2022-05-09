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
#     # save_freq=700
# )

# Create model



color_model = keras.models.load_model(pathlib.Path('D:./TF_backup/finale/model6.h5'))
structure_model = keras.models.load_model(pathlib.Path('D:./TF_backup/finale/cpm.h5'))
others_model = keras.models.load_model(pathlib.Path('D:./TF_backup/finale/other.h5'))
decision_model = keras.models.load_model(pathlib.Path('D:./TF_backup/finale/dt.h5'))
# decision_model.summary()
input_layer = layers.Input(shape=input_shape)
color_model.inputs = input_layer
structure_model.inputs = input_layer
others_model.inputs = input_layer

# print(color_model.outputs.shape)
# print(structure_model.outputs.shape)
# print(others_model.outputs.shape)
inp1 = tf.convert_to_tensor(color_model.outputs)
inp2 = tf.convert_to_tensor(structure_model.outputs)
inp3 = tf.convert_to_tensor(others_model.outputs)

sq1 = tf.squeeze(inp1, axis=0)
sq2 = tf.squeeze(inp2, axis=0)
sq3 = tf.squeeze(inp3, axis=0)

# print(tf.shape(inp1))

# concat = layers.Concatenate(axis=2)([color_model.outputs, structure_model.outputs, others_model.outputs], axis=1)
concat = layers.Concatenate(axis=1)([sq1, sq2, sq3])
# print(tf.shape(concat))

# flat = layers.Reshape((13))(concat)
# print(tf.shape(flat))

# decision_model.inputs = tf.convert_to_tensor(concat)
# print(np.shape(decision_model.inputs))

final_model = keras.Model(input_layer, concat)

# final_model.summary()
# tf.keras.utils.plot_model(final_model, to_file=model_plot_dir, show_shapes=True)
