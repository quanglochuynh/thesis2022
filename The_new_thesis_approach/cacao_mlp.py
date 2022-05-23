import pathlib
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from image_extractor import select_feature

input_shape = (394,)
batch_size = 64
x_train_dir = pathlib.Path('D:/Thesis_data/mlp_data/X_train.npz')
y_train_dir = pathlib.Path('D:/Thesis_data/mlp_data/Y_train.npz')
x_test_dir  = pathlib.Path('D:/Thesis_data/mlp_data/X_test.npz')
y_test_dir  = pathlib.Path('D:/Thesis_data/mlp_data/Y_test.npz')
checkpoint_dir = pathlib.Path('D:./TF_checkpoint/cacao_CNN/weight/')
model_dir = pathlib.Path('D:./TF_backup/mlp/mlp.h5')
model_plot_dir = pathlib.Path('D:./TF_backup/mlp/mlp.png')


y_train = np.asarray(np.load(x_train_dir, allow_pickle=True)['arr_0'], dtype=np.float32)
x_train = np.asarray(np.load(y_train_dir, allow_pickle=True)['arr_0'], dtype=np.float32)
x_test = np.asarray(np.load(x_test_dir, allow_pickle=True)['arr_0'], dtype=np.float32)
y_test = np.asarray(np.load(y_test_dir, allow_pickle=True)['arr_0'], dtype=np.float32)

input_shape = (len(x_train[0]),)

# print(np.shape(x_train))
# print(np.shape(y_train))
# print(np.shape(x_test))
# print(np.shape(y_test))

model_checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_dir,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
)

# # Define model
# input_layer = layers.Input(shape=input_shape)
# den1 = layers.Dense(788, activation='sigmoid')(input_layer)
# drop1 = layers.Dropout(0.1)(den1)
# den2 = layers.Dense(788, activation='sigmoid')(drop1)
# drop2 = layers.Dropout(0.1)(den2)
# den3 = layers.Dense(448, activation='sigmoid')(drop2)
# drop3 = layers.Dropout(0.1)(den3)
# den4 = layers.Dense(14, activation='sigmoid')(drop3)

# model = keras.Model(input_layer, den4)

# opt = tf.keras.optimizers.SGD(
#     learning_rate=0.001,
#     momentum=0.0,
#     nesterov=True,
#     name='SGD',
# )

# model.compile(
#     optimizer=opt, 
#     loss="categorical_crossentropy", 
#     metrics=["accuracy"]
#     )
model = keras.models.load_model(model_dir)

model.summary()

epochs = 100
model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=epochs, callbacks=[model_checkpoint])
model.save(model_dir)

score = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
