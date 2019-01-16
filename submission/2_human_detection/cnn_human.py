#!/bin/python3.6

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from time import time


# Image size (one axis)
size = 128
# Image format (width x height x rgb)
img_format = (128, 128, 3)

# Create sequential model
model = Sequential()

# Convolutional layers (filter-size = 3x3)
model.add(Conv2D(32, 3, activation='relu', input_shape=img_format))
model.add(MaxPooling2D(2))

model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPooling2D(2))

model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPooling2D(2))

model.add(Conv2D(128, 3, activation='relu'))
model.add(MaxPooling2D(2))

# Flatten convolutional data
model.add(Flatten())
# Dropout-layer for not overfitting
model.add(Dropout(0.5))

# Hidden layers
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
train_data = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
valid_data = ImageDataGenerator(1./255)

# Training generators (classes will be automatically generated from subgategories)
train_generator = train_data.flow_from_directory('./data/train/', target_size=(size, size),
                                                 batch_size=64, class_mode='binary')
valid_generator = valid_data.flow_from_directory('./data/valid/', target_size=(size, size),
                                                 batch_size=64, class_mode='binary')

# Initialize tensorboard for visualization (can be called by: "tensorboard --logdir=logs/")
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Fit the model
model.fit_generator(train_generator, epochs=5, steps_per_epoch=63, validation_data=valid_generator, validation_steps=7,
                    workers=4, callbacks=[tensorboard])

# Save model to file
model.save('model_1.h5')
