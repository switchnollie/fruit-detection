import numpy as np
from keras.datasets import cifar100
import scipy.misc
from matplotlib import pyplot as plt


# Load apple, orange, pear and man from CIFAR100-dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

indices_train, indices_test = [], []

# Wir untersuchen, ob es sich bei einem Eintrag der Trainings-Labels um eines der Früchte handelt.
# Falls ja, fügen wir dessen Index einem zuvor initialisiertem Array hinzu.
for i in range(len(y_train)):
    if y_train[i] == 0 or y_train[i] == 53 or y_train[i] == 57:
        indices_train.append(i)

# Selbiges führen wir für alle Test-Labels durch.
for i in range(len(y_test)):
    if y_test[i] == 0 or y_test[i] == 53 or y_test[i] == 57:
        indices_test.append(i)

# Wir reduzieren unsere Trainings- und Test-Labels auf alle die, der Früchte.
y_train = np.array(y_train[indices_train])
y_test = np.array(y_test[indices_test])

# Wir reduzieren unsere Trainings- und Testdaten auf alle die, der Früchte.
x_train = x_train[np.ravel(indices_train)]
x_test = x_test[np.ravel(indices_test)]

# Für die Konvertierung unserer Label-Vektoren in Binäre-Klassenmatrizen, ändern wir alle ursprünglichen
# Trainings- und Test-Labels in die Werte 0-2. Man beachte: range(start, ende) inkludiert ende nicht!
for i in range(len(y_train)):
    if y_train[i] == 0:
        np.put(y_train, i, 0)
    elif y_train[i] == 53:
        np.put(y_train, i, 1)
    elif y_train[i] == 57:
        np.put(y_train, i, 2)

for i in range(len(y_test)):
    if y_test[i] == 0:
        np.put(y_test, i, 0)
    elif y_test[i] == 53:
        np.put(y_test, i, 1)
    elif y_test[i] == 57:
        np.put(y_test, i, 2)


def save_train_images(x_train):
    j = 0
    for j in range(len(x_train)):
        scipy.misc.imsave('../img/train/' + str(j) + '.jpg', x_train[j])


def save_test_images(x_test):
    j = 0
    for j in range(len(x_test)):
        scipy.misc.imsave('../img/test/' + str(j) + '.jpg', x_test[j])


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
