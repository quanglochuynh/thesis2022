import numpy as np
import tensorflow as tf
from tensorflow import keras

dataset = keras.preprocessing.image_dataset_from_directory(
  'D:/Thesis_data/Augmented_224x224', batch_size=64, image_size=(224, 224))

# print(dataset.shape)

# for data, labels in dataset:
#    print(data.shape)  # (64, 200, 200, 3)
#    print(data.dtype)  # float32
#    print(labels.shape)  # (64,)
#    print(labels.dtype)  # int32