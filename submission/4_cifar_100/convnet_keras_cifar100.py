import numpy as np

from keras.datasets import cifar100
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

from sklearn.metrics import confusion_matrix

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
y_test_save = y_test

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
# apple.set_title("Apple")
# apple.imshow(x_train[0])
#
# orange = fig.add_subplot(1, 3, 2)
# orange.set_title("Orange")
# orange.imshow(x_train[1])
#
# pear = fig.add_subplot(1, 3, 3)
# pear.set_title("Pear")
# pear.imshow(x_train[6])
#
# plt.show()

# Wir müssen zuvor unsere Bilddaten in Float-Datentypen casten, dass wir Gleitkommazahlen bekommen.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Konvertieren der RGB-Werte in Werte zwischen [0;1]
x_train /= 255
x_test /= 255

# Wir definieren unser Model über einen Stapel von Schichten. Hierzu initialisieren wir unser sequentialles Modell.
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(96, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(192, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.4))
model.add(Flatten())

model.add(Dense(96, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

# Vorhersagen der Testdaten-Labels.
y_pred = model.predict_classes(x_test, verbose=0)
# Umkehren der binären Klassenmatrix zu kategorischen Vektoren für Confusion Matrix.
y_test_rev = [np.argmax(y, axis=None, out=None) for y in y_test]

print(y_test_rev)
print(confusion_matrix(y_test_rev, y_pred))

# Save model to file
model.save("model11.h5")

# Epochs: 10 (Layers: 32, 64, 128, dropout:0.4, 64, 3)
# loss: 0.4563 - acc: 0.8133
# val_loss: 0.4367 - val_acc: 0.8467
# Okay live classification (pear, orange but not apple)

# Epochs: 20 "
# loss: 0.2474 - acc: 0.9053
# val_loss: 0.3856 - val_acc: 0.8467
# Poor live classification

# Epochs: 20 (Layers: 32, 96, 192, dropout:0.3, 96, 3)
# loss: 0.0995 - acc: 0.9660
# val_loss: 0.4565 - val_acc: 0.8600
# Extreme bad live classification

# Epochs: 20 (Layers: 96, 192, 384, dropout:0.5, 1920, 3)
# loss: 0.2648 - acc: 0.8987
# val_loss: 0.4645 - val_acc: 0.8567

# Epochs: 20 (Layers: 32, 64, 128+128, dropout:0.2, 96, 3)
# loss: 0.1779 - acc: 0.9420
# val_loss: 0.5365 - val_acc: 0.8400

# Epochs: 20 (Layers: 32, 96, 256+256, dropout:0.2, 2046, 3)
# loss: 0.1328 - acc: 0.9553
# val_loss: 0.5348 - val_acc: 0.8533

# Epochs: 20 (Layers: 96, 288, 864, dropout:0.3, 2592, 3)
# loss: 0.2107 - acc: 0.9220
# val_loss: 0.5102 - val_acc: 0.8300

# Epochs: 20 (Layers: 32, 96, 192, dropout:0.3, 96, 3)
# loss: 0.0770 - acc: 0.9720
# val_loss: 0.3765 - val_acc: 0.8867
# model10
