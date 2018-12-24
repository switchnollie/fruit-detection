import numpy as np

from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Sequential
from keras.datasets import mnist

# Set image input shape needed for first layer
img_shape = (28, 28, 1)
# Create a sequential model
model = Sequential()
# Add a convolutional layer 3, 3 by 3 filters (usually called kernel) and a stride size of 1 and add activation function
# Kernel size = 2x2 matrix
model.add(Conv2D(6, (2, 2), input_shape=img_shape, activation='relu'))
# Add a pooling layer
model.add(MaxPool2D(2))
# Flatten convolutional layers for output layer
model.add(Flatten())
# Add Dense layer for classification and add activation function
model.add(Dense(10, activation='softmax'))
# Print model summary
model.summary()
# Training consisting of optimizer, loss function and metric
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Read MNIST-data (handwritten digits)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Rotate dataset to get vertical array (4 dimensions instead of 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
# # Train model on training data add validation_data=(x_test, y_test) for evaluation after each epoch
model.fit(x_train, y_train, batch_size=32, epochs=10)
# Save model weights
# model.save_weights("cnn_weights.h5")
# Evaluate model after training on test data
score = model.evaluate(x_test, y_test, verbose=1)
print("Accuracy on test-data: " + str(score[1] * 100))
