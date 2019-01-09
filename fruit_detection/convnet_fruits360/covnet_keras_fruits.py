#!/bin/python3.6

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from time import time


class KerasModels:

    @staticmethod
    def create_cnn_model():

        img_width = 100
        img_height = 100
        img_channels = 3

        # Create keras sequential model
        model = Sequential()

        # Convolutional layers (filter-size = 3x3)
        model.add(Conv2D(32, 3, activation='relu', input_shape=(img_width, img_height, img_channels)))
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
        model.add(Dropout(0.4))

        # Hidden layers
        model.add(Dense(512, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        # Compile model
        # Could try categorical_accuracy
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        train_data = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        valid_data = ImageDataGenerator(1. / 255)

        # Training generators (classes will be automatically generated from subgategories)
        train_generator = train_data.flow_from_directory('../dataset/train/',
                                                         target_size=(img_width, img_height),
                                                         batch_size=75, class_mode='categorical')
        valid_generator = valid_data.flow_from_directory('../dataset/valid/',
                                                         target_size=(img_width, img_height),
                                                         batch_size=75, class_mode='categorical')

        # Initialize tensorboard for visualization (can be called by: "tensorboard --logdir=logs/")
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        # Fit the model
        model.fit_generator(train_generator, epochs=5, steps_per_epoch=24, validation_data=valid_generator,
                            validation_steps=8, workers=4, callbacks=[tensorboard])

        # Save model to file
        model.save("model" + str(time()) + ".h5")

        return model.summary()
