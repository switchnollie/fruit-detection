from keras.datasets import cifar100
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from time import time
import numpy as np


# Load apple, orange, pear and man from CIFAR100-dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

indices_train = []
indices_test = []

# 46 = man
for i in range(len(y_train)):
    if y_train[i] == 0 or y_train[i] == 46 or y_train[i] == 53 or y_train[i] == 57:
        indices_train.append(i)

for i in range(len(y_test)):
    if y_test[i] == 0 or y_test[i] == 46 or y_test[i] == 53 or y_test[i] == 57:
        indices_test.append(i)

y_train = np.array(y_train[indices_train])
y_test = np.array(y_test[indices_test])

x_train = x_train[np.ravel(indices_train)]
x_test = x_test[np.ravel(indices_test)]

for i in range(len(y_train)):
    if y_train[i] == 46:
        np.put(y_train, i, 0)
    elif y_train[i] == 0:
        np.put(y_train, i, 1)
    elif y_train[i] == 53:
        np.put(y_train, i, 2)
    elif y_train[i] == 57:
        np.put(y_train, i, 3)

for i in range(len(y_test)):
    if y_test[i] == 46:
        np.put(y_test, i, 0)
    elif y_test[i] == 0:
        np.put(y_test, i, 1)
    elif y_test[i] == 53:
        np.put(y_test, i, 2)
    elif y_test[i] == 57:
        np.put(y_test, i, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Needs to be used for categorical_accuracy-metrics
y_train = to_categorical(y_train, 4)
y_test = to_categorical(y_test, 4)

# Create convnet
model = Sequential()
model.add(Conv2D(96, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.4))
model.add(Flatten())

model.add(Dense(2048, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
# Save model to file
model.save("model" + str(time()) + ".h5")
