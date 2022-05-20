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
image = cv2.imread('D:/Thesis_data/mlp_data/training_img/Plated_Purple/image(6).JPG')
# cv2.imshow('abc', image)
ft1, ft2, glcm1, glcm2, glcm3, glcm4 = select_feature(image)

