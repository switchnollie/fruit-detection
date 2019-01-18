#!/usr/bin/env python
# coding: utf-8

# 

# # Fruit Detection

# Für den ersten Versuch eines neuronalen Netzes zur Klassifizierung der Bilder soll ein Multi Layer Perceptron Netz in Keras implementiert und evaluiert werden. Aufgrund seiner geringen Komplexität im Vergleich zu anderen Ansätzen bietet dies einen vergleichsweise einfachen Einstieg in die Bildklassifizierung. Außerdem können hierbei die in der Vorlesung behandelten Konzepte angewandt werden.

# ## Bibliotheken
# 
# Wir verwenden die `tf.keras` Library für das neuronale Netz (`model`). Keras bietet eine sehr hohe Abstraktion ("High Level API") auf die Deep Learning Modelle, weshalb es für Einsteiger eine sehr populäre Option darstellt. Des weiteren ermöglicht Keras mit wenig Code ein Modell auszuprobieren und somit schnell einen Ansatz zu evaluieren oder einen Prototypen zu erstellen.
# `tf.keras` ist die in Tensorflow integrierte Version von Keras und setzt somit auf der Core API von Tensorflow auf. 
# 
# In unserem Prototyp trainieren und evaulieren wir ein `Sequential` Modell, also einen linearen Stack aus Layern.
# 
# Als Datenstruktur für die Bilder und später die Tensoren verwenden wir die n-dimensionalen `numpy`-Arrays, da diese mit Fancy-Indexing, der `.shape` Property und den komponentenweise Operationen eine intuitive Arbeitsweise mit Matrizen ermöglicht.
# 
# Für die Visualisierung der Ergebnisse setzen wir auf eine Kombination aus `matplotlib` und `Tensorboard`, wobei matplotlib für die Visualisierung der gelabelten Eingabe-Daten und für eine statische Visualisierung der Ergebnisse benutzt wird und Tensorboard dynamische, interaktive Visualisierungen auch über mehrere Ausführungen hinweg generiert. Tensorboard dient entsprechent nicht nur der Dokumentation, sondern auch in hohem Maße der Optimierung des Modells.
# 

# In[1]:


import numpy as np
import tensorflow as tf
from time import time

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation
from tensorflow.image import rgb_to_grayscale

import matplotlib.pyplot as plt


# ## Vorverarbeitung der Daten
# 
# ### Herunterladen und Split in Trainings- und Testdatensatz
# 
# Der cifar100-Datensatz wird mit der `load_data()`-Methode aus dem Archiv des Canadian Institute For Advanced Research heruntergeladen, entpackt und in der Form von 2 Tupeln für die gelablten Trainings- und Testdaten zurückgegeben. Der Datensatz setzt sich aus Kategorien samt Subkategorien zusammen. Über das Argument label_mode fordern wir alle Subkategorien an. 
# Diese Trainings- bzw- Test-Tupel bestehen wiederum selbst aus 2 uint8-Numpy-Arrays der shape (50000, 32, 32, 3). Die Bilder sind also 32x32 Pixel groß und haben 3 Farbkanäle (R,G,B). 
# 
# Die Labels sind ebenfalls uint8-Numpy-Arrays, allerdings nur 2-dimensional (50000 Labels mit je einem Index, der zu einer bestimmten Bildklasse gehört).
# 
# Mit Hilfe der Unpacking Syntax lassen sich die Rückgabe-Tupel in ihre Numpy-Array Bestandteile destrukturieren und diese direkt einer Variablen zuweisen. Es werden also 4 separate Variablen mit nur einer Codezeile definiert und initialisiert.

# In[2]:


(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
print("Image Data Shape:", x_train.shape)
print("Data Type of Images:", type(x_train))
print("Labels Shape:", y_train.shape)


# ### Filtern der relevanten Klassen
# 
# Da wir mit unserem Modell nur **Früchte** klassizieren möchten, reduzieren wir unsere Daten auf die Kategorien: **Apple**, **Orange**, **Pear**. Diese Klassen entsprechen den Labels mit folgenden Indices: Apple = 0, Orange = 53, Pear = 57.

# In[3]:


# Wir filtern die Numpy Arrays mittels Numpy Fancy Indexing nach Indices 
# der Werte 0, 53 oder 57
indices_train = np.argwhere((y_train == 0) | (y_train == 53) | (y_train == 57))[:,0]
indices_test = np.argwhere((y_test == 0) | (y_test == 53) | (y_test == 57))[:,0]
# Wir reduzieren unsere Trainings- und Test-Labels auf die Einträge mit Früchten
y_train = np.array(y_train[indices_train])
y_test = np.array(y_test[indices_test])
# Selbiges tun wir für die Trainings- und Testdaten.
x_train = x_train[np.ravel(indices_train)]
x_test = x_test[np.ravel(indices_test)]

# Nun müssen wir nur noch die Werte auf 0, 1 und 2 mappen, dann entsprechen
# die Werte den Indize einer Liste der Länge 3

def mapLabels(y):
  if y == 53: return [1]
  elif y == 57: return [2]
  else: return [0]
  
y_train = np.array(list(map(mapLabels, y_train[:,0].tolist())))
y_test = np.array(list(map(mapLabels, y_test[:,0].tolist())))

# Die Liste beinhaltet die "menschlich lesbaren Kategorien", das Netz arbeitet
# ausschließlich mit den Indices.

class_names = ["Apfel", "Orange", "Birne"]


# In[4]:


# Convert to grayscale
x_train = tf.Session().run(rgb_to_grayscale(
    x_train,
    name=None
))
x_test = tf.Session().run(rgb_to_grayscale(
    x_test,
    name=None
))

# Drop last dimension
x_train = x_train[:,:,:,0]
x_test = x_test[:,:,:,0]

print(x_train[0].shape)


# In[ ]:


x_train = x_train / 255.0

x_test = x_test / 255.0


# In[ ]:


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.asscalar(y_train[i])])


# In[ ]:



model = Sequential([
    Flatten(input_shape=(32, 32)),
    Dense(1024, activation=tf.nn.relu),
    Dense(3, activation=tf.nn.softmax) 

])


# In[ ]:


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


tensorboardCb = TensorBoard(log_dir="logs/{}".format(time()))


# In[ ]:


history = model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test))


# In[ ]:


test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)


# In[ ]:


predictions = model.predict(x_test)


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('acc/loss')
plt.xlabel('epoch')
plt.legend(['train_acc', 'test_acc', 'train_loss', 'test_loss'], loc='best')
plt.show()


# In[ ]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[np.asscalar(predicted_label)],
                                100*np.max(predictions_array),
                                class_names[np.asscalar(true_label)]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i].tolist(), true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(3), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[np.asscalar(predicted_label)].set_color('red')
  thisplot[np.asscalar(true_label)].set_color('blue')


# In[ ]:


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, y_test, x_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, y_test)

