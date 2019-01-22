import numpy as np

from keras.datasets import cifar100
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.callbacks import TensorBoard
from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from time import time

# import matplotlib.pyplot as plt

# Load apple, orange, pear and man from CIFAR100-dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# Wir untersuchen, ob es sich bei einem Eintrag der Trainings-Labels um eines der Früchte handelt.
# Falls ja, fügen wir dessen Index einem zuvor initialisiertem Array hinzu.
# 0 = Apfel, 1 = Orange, 2 = Birne
indices_train = np.where((y_train == 0) | (y_train == 53) | (y_train == 57))[0]
indices_test = np.where((y_test == 0) | (y_test == 53) | (y_test == 57))[0]

# Wir reduzieren unsere Trainings- und Test-Labels auf alle die, der Früchte.
y_train = np.array(y_train[indices_train])
y_test = np.array(y_test[indices_test])
# Wir reduzieren unsere Trainings- und Testdaten auf alle die, der Früchte.
x_train = x_train[indices_train]
x_test = x_test[indices_test]

# Für die Konvertierung unserer Label-Vektoren in Binäre-Klassenmatrizen, ändern wir alle ursprünglichen
# Trainings- und Test-Labels in die Werte 0-2 mittels lambda-Funktionen.
y_train = np.array(list(map(lambda i: [1] if i == 53 else ([2] if i == 57 else [0]), y_train)))
y_test = np.array(list(map(lambda i: [1] if i == 53 else ([2] if i == 57 else [0]), y_test)))

# Die Konvertierung unser Label-Vektoren in Binäre-Klassenmatrizen wird für unseren Datensatz benötigt.
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

# Dafür plotten wir jeweils 1 Bild der drei Kategorien.
# fig = plt.figure(figsize=(16, 6))
#
# apple = fig.add_subplot(1, 3, 1)
# apple.set_title(y_train[0])
# apple.imshow(x_train[0])
#
# orange = fig.add_subplot(1, 3, 2)
# orange.set_title(y_train[1])
# orange.imshow(x_train[1])
#
# pear = fig.add_subplot(1, 3, 3)
# pear.set_title(y_train[6])
# pear.imshow(x_train[6])
#
# plt.show()

# Wir müssen zuvor unsere Bilddaten in Float-Datentypen casten, dass wir Gleitkommazahlen bekommen.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Konvertieren der RGB-Werte in Werte zwischen [0;1]
x_train /= 255
x_test /= 255

# Splitten Trainingsdatensatz in Training und Validierung
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.20)

# Wir definieren unser Model über einen Stapel von Schichten. Hierzu initialisieren wir unser sequentialles Modell.
model = Sequential()

model.add(
    Conv2D(32, 3, activation='relu', data_format="channels_last", input_shape=(32, 32, 3), strides=1, padding="valid",
           kernel_initializer="he_uniform"))
model.add(MaxPooling2D(2))
model.add(Dropout(0.2))

model.add(
    Conv2D(32, 3, activation='relu', data_format="channels_last", input_shape=(32, 32, 3), strides=1, padding="valid",
           kernel_initializer="he_uniform"))
model.add(MaxPooling2D(2))
model.add(Dropout(0.3))

model.add(
    Conv2D(32, 3, activation='relu', data_format="channels_last", input_shape=(32, 32, 3), strides=1, padding="valid",
           kernel_initializer="he_uniform"))
model.add(MaxPooling2D(2))
model.add(Dropout(0.4))

# Dient der Dimensionsreduktion. Wird für Verwendung von Dense Layern benötigt.
model.add(Flatten())

# Voll verbundene Neuronen.
model.add(Dense(1024, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Konfiguration der Trainingsparameter.
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])

model.summary()

# Initialisieren unseres Tensorboard-Callbacks zur späteren Visualisierung unserer Metriken.
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

history = model.fit(x_train, y_train, epochs=50, validation_data=(x_valid, y_valid), callbacks=[tensorboard])

# Vorhersagen der Testdaten-Labels.
y_pred = model.predict_classes(x_test, verbose=0)

# Ausgeben der Test-accuracy
score = model.evaluate(x_test, y_test)
print("Test-accuracy: " + str(score[1] * 100) + "%")

# Umkehren der binären Klassenmatrix zu kategorischen Vektoren für Confusion Matrix.
# Gibt Indice des größten Wertes zurück.
y_test_rev = [np.argmax(y, axis=None, out=None) for y in y_test]
print(confusion_matrix(y_test_rev, y_pred))

# Accuracy und loss für train und test plotten.
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.ylabel("accuracy / loss")
plt.xlabel("epoch")
plt.legend(["train_acc", "test_acc", "train_loss", "test_loss"], loc="center right")
plt.rcParams["figure.figsize"] = (35, 20)
plt.show()

# # Save model to file
# model.save("../h5/model_9_100.h5")
