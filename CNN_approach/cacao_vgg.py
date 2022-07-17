import pathlib
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications.vgg16 import VGG16

import numpy as np
import matplotlib.pyplot as plt

img_height = 224
img_width = 224
input_shape = (img_width, img_height, 3)
batch_size = 16
train_dir = pathlib.Path('D:/Thesis_data/final_dataset/training_img')
test_dir  = pathlib.Path('D:/Thesis_data/final_dataset/testing_img')
checkpoint_dir = pathlib.Path('D:./TF_checkpoint/vgg16/weight/')
model_dir = pathlib.Path('D:./TF_backup/vgg16/vgg16.h5')
model_plot_dir = pathlib.Path('D:./TF_backup/vgg16/vgg16.png')


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

model_checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_dir,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
)

base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
base_model.trainable = False
# base_model.summary()
# tf.keras.utils.plot_model(base_model, to_file=model_plot_dir, show_shapes=True)
input_layer = layers.Input(shape=input_shape)
vgg = base_model(input_layer, training = False)
flat = layers.Flatten()(vgg)
x = layers.Dropout(0.2)(flat)
x = layers.Dense(14)(x)
new_vgg = keras.Model(input_layer, x)
new_vgg.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# new_vgg.summary()

epochs = 4
new_vgg.fit(train_ds, epochs=epochs, callbacks=[model_checkpoint])
new_vgg.save(model_dir)


# Test model
test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

score = new_vgg.evaluate(test_ds, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# draw confusion matrix
i = 0
cmatrix = np.zeros((4,4), dtype=np.int16)
for img,lab in test_ds.take(20):
    result = new_vgg.predict(img, use_multiprocessing=True)
    for j in range(np.size(result,axis=0)):
        id1 = np.argmax(result[j])
        id2 = np.argmax(lab[j])
        # print(result[j], lab[j].numpy())
        cmatrix[id1][id2] = cmatrix[id1][id2] + 1
cmatrix = np.absolute(cmatrix)
print(repr(cmatrix))

fig = plt.figure(figsize=(6,6))
plt.imshow(cmatrix)
plt.title("Plot 2D array")
plt.show()