import pathlib
import tensorflow as tf
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
# from image_extractor import select_feature

input_shape = (394,)
batch_size = 16
x_train_dir = pathlib.Path('D:/Thesis_data/mlp_data/backups/X_train.npz')
y_train_dir = pathlib.Path('D:/Thesis_data/mlp_data/backups/Y_train.npz')
x_test_dir  = pathlib.Path('D:/Thesis_data/mlp_data/backups/X_test.npz')
y_test_dir  = pathlib.Path('D:/Thesis_data/mlp_data/backups/Y_test.npz')
checkpoint_dir = pathlib.Path('D:./TF_checkpoint/cacao_CNN/weight/')
model_dir = pathlib.Path('D:./TF_backup/mlp/mlp.h5')
model_plot_dir = pathlib.Path('D:./TF_backup/mlp/mlp.png')


y_train = np.asarray(np.load(x_train_dir, allow_pickle=True)['arr_0'], dtype=np.float32)
x_train = np.asarray(np.load(y_train_dir, allow_pickle=True)['arr_0'], dtype=np.float32)
x_test = np.asarray(np.load(x_test_dir, allow_pickle=True)['arr_0'], dtype=np.float32)
y_test = np.asarray(np.load(y_test_dir, allow_pickle=True)['arr_0'], dtype=np.float32)

input_shape = (len(x_train[0]),)

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))

model_checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_dir,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
)

# normalizer = layers.Normalization()
# normalizer.adapt(x_train)

# # Define model
# input_layer = layers.Input(shape=input_shape)
# norm = normalizer(input_layer)
# den1 = layers.Dense(512, activation='relu', kernel_initializer='HeNormal')(norm)
# drop1 = layers.Dropout(0.2)(den1)
# den2 = layers.Dense(512, activation='relu', kernel_initializer='HeNormal')(drop1)
# drop2 = layers.Dropout(0.2)(den2)
# den3 = layers.Dense(256, activation='relu', kernel_initializer='HeNormal')(drop2)
# drop3 = layers.Dropout(0.2)(den3)
# den4 = layers.Dense(14, activation='softmax')(drop3)
# model = keras.Model(input_layer, den4)

# opt = tf.keras.optimizers.SGD(
#     learning_rate=0.0001,
#     nesterov=False,
#     name='SGD',
# )

# model.compile(
#     optimizer=opt, 
#     loss="categorical_crossentropy", 
#     metrics=["accuracy"]
#     )

model = keras.models.load_model(model_dir)
model.summary()

epochs = 20
# model.fit(x_train, y_train, batch_size=batch_size, shuffle=True, epochs=epochs, callbacks=[model_checkpoint])
# model.save(model_dir)

# score = model.evaluate(x_test, y_test, verbose=1)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])

# draw confusion matrix
import sys
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

i = 0
cmatrix = np.zeros((14,14), dtype=np.int16)
for i in range(len(x_test)):
    progress(i, 1680)
    result = model.predict(np.array([x_test[i]])).flatten()
    id1 = np.argmax(result)
    id2 = np.argmax(y_test[i])
    cmatrix[id1][id2] = cmatrix[id1][id2] + 1
    # break

cmatrix = np.absolute(cmatrix)
print(repr(cmatrix))

fig = plt.figure(figsize=(6,6))
plt.imshow(cmatrix)
plt.title("Plot 2D array")
plt.show()