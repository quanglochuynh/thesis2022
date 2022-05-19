import pathlib
import cv2
# from keras import layers
# import numpy as np
# import matplotlib.pyplot as plt
from image_extractor import select_feature

# input_shape = (img_width, img_height, 3)
batch_size = 16
train_dir = pathlib.Path('D:/Thesis_data/mlp_data/training_img')
test_dir  = pathlib.Path('D:/Thesis_data/mlp_data/testing_img')
checkpoint_dir = pathlib.Path('D:./TF_checkpoint/cacao_CNN/weight/')
model_dir = pathlib.Path('D:./TF_backup/mlp/mlp.h5')
model_plot_dir = pathlib.Path('D:./TF_backup/mlp/mlp.png')

# im_dir = pathlib.Path. (train_dir, )
image = cv2.imread('D:/Thesis_data/mlp_data/training_img/Brittle/image(1).JPG')
# cv2.imshow('abc', image)
print(select_feature(image))
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

# #MODEL
# input_layer = layers.Input(shape=input_shape)
# dense1 = layers.Dense()

# cv2.waitKey(0)
# cv2.destroyAllWindows()