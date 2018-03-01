import numpy as np
from matplotlib import pyplot as pp
import keras
import h5py

import nn_functions as nn_f
import unpack


X_train, Y_train = unpack.import_training_data()
X_dev, Y_dev = unpack.import_test_data()

Y_train = nn_f.one_hot_encoding(Y_train, 10)
Y_dev = nn_f.one_hot_encoding(Y_dev, 10)

X_train = X_train.reshape((50000, 32, 32, 3), order = 'F')
X_dev = X_dev.reshape((10000, 32, 32, 3), order = 'F')

model = nn_f.ResNet(input_shape = (32, 32, 3), classes = 10)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs = 1, batch_size = 100)

model.save('cifar_model.h5')

preds = model.evaluate(X_dev, Y_dev)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))