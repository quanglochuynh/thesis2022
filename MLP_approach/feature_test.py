from cgi import test
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from image_extractor import DataSetup
data = DataSetup()

checkpoint_dir = 'D:./TF_checkpoint/mlp/weight/'
model_checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_dir,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
)

batch_size = 8

class Comb:
    def __init__(self, ft):
        self.data = ft

test_list = []
test_list.append(Comb([7]))
test_list.append(Comb([0]))
test_list.append(Comb([0,1]))
test_list.append(Comb([0,3]))
test_list.append(Comb([0,5]))
test_list.append(Comb([0,5,6]))
test_list.append(Comb([8,9,10]))
test_list.append(Comb([0,5,6,7]))
test_list.append(Comb([0,5,6,8,9,10]))
test_list.append(Comb([0,3,5,6,8,9,10]))
test_list.append(Comb([0,1,3,5,6,8,9,10]))
test_list.append(Comb([0,1,3,4,5,6,8,9,10]))         

# for i in range(11,11):
i=11
data.concat(dataID=test_list[i].data)
model_dir = 'D:./TF_backup/mlp/' + data.model_name
print(i,"-",model_dir)
input_shape = (data.length,)

normalizer = layers.Normalization()
normalizer.adapt(data.x_train)
# Define model
input_layer = layers.Input(shape=input_shape)
norm = normalizer(input_layer)
den1 = layers.Dense(1024, activation='tanh', kernel_initializer='HeNormal')(norm)
drop1 = layers.Dropout(0.2)(den1)
den2 = layers.Dense(1024, activation='tanh', kernel_initializer='HeNormal')(drop1)
drop2 = layers.Dropout(0.2)(den2)
den3 = layers.Dense(512, activation='tanh', kernel_initializer='HeNormal')(drop2)
drop3 = layers.Dropout(0.2)(den3)
den4 = layers.Dense(512, activation='tanh', kernel_initializer='HeNormal')(drop3)
drop4 = layers.Dropout(0.2)(den4)
den5 = layers.Dense(14, activation='softmax')(drop4)
model = keras.Model(input_layer, den5)

opt = tf.keras.optimizers.SGD(
    learning_rate=0.0001,
    momentum=0.4,
    nesterov=True,
    name='SGD',
)

model.compile(
    optimizer=opt, 
    loss="categorical_crossentropy", 
    metrics=["accuracy"]
    )
accu = np.array([0])
history = model.fit(data.x_train, data.y_train, batch_size=16, shuffle=True, epochs=20, callbacks=[model_checkpoint])
accu = np.concatenate([accu, history.history['accuracy']],axis=None)
history = model.fit(data.x_train, data.y_train, batch_size=8, shuffle=True, epochs=20, callbacks=[model_checkpoint])
accu = np.concatenate([accu, history.history['accuracy']],axis=None)
history = model.fit(data.x_train, data.y_train, batch_size=4, shuffle=True, epochs=20, callbacks=[model_checkpoint])
accu = np.concatenate([accu, history.history['accuracy']],axis=None)


score = model.evaluate(data.x_test, data.y_test, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

max_accu = np.max(accu)
model_dir = model_dir + "_" + str(round(max_accu,4)) + "_" + str(round(score[1],4))
model.save(model_dir + ".h5")
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.grid(True, which='both')
plt.ylim(0,1)
plt.legend(["Training accuracy", "Training loss"])
plt.rcParams["legend.loc"] ='lower right'
plt.savefig(model_dir + ".png")