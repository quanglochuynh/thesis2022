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
train_dir = pathlib.Path('D:/Thesis_data/color_training_img')
test_dir  = pathlib.Path('D:/Thesis_data/color_testing_img')
checkpoint_dir = pathlib.Path('D:./TF_checkpoint/cacao_CNN/weight/')
model_dir = pathlib.Path('D:./TF_backup/mlp/mlp.h5')
model_plot_dir = pathlib.Path('D:./TF_backup/mlp/mlp.png')