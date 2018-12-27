#!/bin/python3.6

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import v_measure_score

# Random seed for reproduction
values, labels = make_moons(random_state=1)

v_train, v_test, l_train, l_test = train_test_split(values, labels, test_size=0.5)

# # Plot dataset
# plt.title('Make_moons-Dataset')
# plt.scatter(values[:, 0], values[:, 1], marker='o', c=labels)
# plt.show()
# # Print unique labels
# print(len(values))
# print(np.unique(labels))

# Create model
model = Sequential()
# Add first layer (input layer) gets 2 inputs and uses "relu" activation function (better performance than sigmoid)
model.add(Dense(32, input_dim=2, activation='relu'))
# Hidden layers don't need input-dimensions
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
# Last layer (output layer) has 1 node/neuron to predict binary classification result
model.add(Dense(1, activation='sigmoid'))

# Compile (configure training process) model. Metrics-value is defined as accuracy (default for classification problems)
model.compile(loss='binary_crossentropy', optimizer='AdaDelta', metrics=['accuracy'])

# Fit the model (we can implement train_test_split to test model on test-data afterwards)
model.fit(v_train, l_train, epochs=200, batch_size=32)    # Not sure if epochs=150 works (we only have 100 value-sets)

# Evaluate the model
score = model.evaluate(v_test, l_test)
print("Accuracy: " + str(score[1] * 100))

# Predict test-data
# l_predict = model.predict_classes(v_test, verbose=1)
# Evaluate prediction


# l_predict_flatten = l_predict.flatten()
# print("Prediction accuracy: " + str(round(v_measure_score(l_test, l_predict_flatten) * 100)) + "%")
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = v_train[:, 0].min() - .5, v_train[:, 0].max() + .5
    y_min, y_max = v_train[:, 1].min() - .5, v_train[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    print(Z.shape, xx.shape)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z)
    plt.scatter(v_train[:, 0], v_train[:, 1], c=l_train)


plot_decision_boundary(model.predict_classes)
plt.show()
