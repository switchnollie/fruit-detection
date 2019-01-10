import numpy as np
from keras.datasets import cifar100
import scipy.misc
from matplotlib import pyplot as plt


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

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')


def save_train_images(x_train):
    j = 0
    for j in range(len(x_train)):
        scipy.misc.imsave('./images/train/' + str(j) + '.jpg', x_train[j])


def save_test_images(x_test):
    j = 0
    for j in range(len(x_test)):
        scipy.misc.imsave('./images/test/' + str(j) + '.jpg', x_test[j])


# save_train_images(x_train)
# save_test_images(x_test)

def show_image_plots(x_train):
    # Dafür plotten wir jeweils 1 Bild der drei Kategorien.
    fig = plt.figure(figsize=(16, 6))

    apple = fig.add_subplot(1, 3, 1)
    apple.set_title("Apple")
    apple.imshow(x_train[0])

    orange = fig.add_subplot(1, 3, 2)
    orange.set_title("Orange")
    orange.imshow(x_train[1])

    pear = fig.add_subplot(1, 3, 3)
    pear.set_title("Pear")
    pear.imshow(x_train[6])

    plt.show()


# show_image_plots(x_train)
