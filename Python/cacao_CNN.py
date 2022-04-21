import pathlib
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

img_height = 224
img_width = 224
input_shape = (img_width, img_height, 3)
batch_size = 8
train_dir = pathlib.Path('D:/Thesis_data/color_training_img')
test_dir  = pathlib.Path('D:/Thesis_data/color_testing_img')
checkpoint_dir = pathlib.Path('D:./TF_checkpoint/cacao_CNN/weight/')
model_dir = pathlib.Path('D:./TF_backup/cacao_CNN/model1_65%.h5')

train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    seed=665,
    shuffle=True,
    validation_split=0.3,
    subset="training",
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
    # save_freq=700
)

## Create model
input_layer = layers.Input(shape=input_shape)
input_layer = layers.Rescaling(scale=1./255, offset=0)(input_layer)
conv1 = layers.Conv2D(28, kernel_size=(7,7), activation="relu")(input_layer)
pool1 = layers.MaxPooling2D(pool_size=(4,4), strides=4)(conv1)
conv2 = layers.Conv2D(14, kernel_size=(5,5), activation="relu")(pool1)
pool2 = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv2)
conv3 = layers.Conv2D(7, kernel_size=(3,3), activation="relu")(pool2)
pool3 = layers.MaxPooling2D(pool_size=(2,2), strides=2)(conv3)
flat = layers.Flatten()(pool3)
dense2 = layers.Dense(196, activation="relu")(flat)
dense3 = layers.Dense(96, activation="relu")(dense2)
dense4 = layers.Dense(32, activation="relu")(dense3)
output_layer = layers.Dense(6, activation="softmax")(dense4)

model = keras.Model(input_layer, output_layer)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.summary()

# Load model
# model.load_weights(checkpoint_dir)
# model = keras.models.load_model(model_dir)
model.summary()

## Train model
epochs = 8
model.fit(train_ds, epochs=epochs, callbacks=[model_checkpoint])
model.save(model_dir)


## Test model
test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    shuffle=True,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

score = model.evaluate(test_ds, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

## draw confusion matrix
i = 0
cmatrix = np.zeros((6,6), dtype=np.int8)
for img,lab in test_ds.take(1024):
    i = i+1
    print(i)
    result = model.predict(img, use_multiprocessing=True)
    for j in range(batch_size):
        id1 = np.argmax(result[j])
        id2 = np.argmax(lab[j])
        # print(id1, id2)
        cmatrix[id1][id2] = cmatrix[id1][id2] + 1
cmatrix = np.absolute(cmatrix)
print(repr(cmatrix))

fig = plt.figure(figsize=(6,6))
plt.imshow(cmatrix)
plt.title("Plot 2D array")
plt.show()