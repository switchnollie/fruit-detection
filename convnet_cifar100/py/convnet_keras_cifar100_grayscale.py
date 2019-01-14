import numpy as np

from keras.datasets import cifar100
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.callbacks import TensorBoard

import tensorflow as tf

from sklearn.metrics import confusion_matrix

from time import time

# import matplotlib.pyplot as plt

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

# Convert image-data into grayscale
x_train = tf.Session().run(tf.image.rgb_to_grayscale(x_train))
x_test = tf.Session().run(tf.image.rgb_to_grayscale(x_test))

# Konvertieren der RGB-Werte in Werte zwischen [0;1]
x_train /= 255
x_test /= 255

# Wir definieren unser Model über einen Stapel von Schichten. Hierzu initialisieren wir unser sequentialles Modell.
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(192, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Ein Dropout-Layer dient der Verhinderung eines Overfittings während des Trainungsdurch zufällige Deaktivierung
# von Neuronen (hier: 40%).
model.add(Dropout(0.4))
# Dient der Dimensionsreduktion. Wird für Verwendung von Dense Layern benötigt.
model.add(Flatten())

# Voll verbundene Neuronen.
model.add(Dense(96, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Konfiguration der Trainingsparameter.
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Initialisieren unseres Tensorboard-Callbacks zur späteren Visualisierung unserer Metriken.
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[tensorboard])

# Vorhersagen der Testdaten-Labels.
y_pred = model.predict_classes(x_test, verbose=0)

# Ausgeben der Test-accuracy
score = model.evaluate(x_test, y_test)
print("Test-accuracy: " + str(score[1]*100) + "%")

# Umkehren der binären Klassenmatrix zu kategorischen Vektoren für Confusion Matrix.
# Gibt Indice des größten Wertes zurück.
y_test_rev = [np.argmax(y, axis=None, out=None) for y in y_test]
print(confusion_matrix(y_test_rev, y_pred))

# Save model to file
model.save("model_gr.h5")
