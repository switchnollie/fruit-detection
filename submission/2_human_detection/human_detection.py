#!/bin/python3.6

import cv2
import numpy as np
from PIL import Image
from keras import models

# Image size (one axis)
size = 128
# Initialize person counter
counter = pred = 0

# Load saved cnn-model
model = models.load_model('model.h5')

# Capture onboard webcam
cam = cv2.VideoCapture(0)

# Show webcam and predictions
while True:

    # Read in webcam frame...
    _, frame = cam.read()

    # ... in RGB-mode
    img = Image.fromarray(frame, 'RGB')
    # Resize img for accuracy in prediction
    img = img.resize((size, size))
    # Create an array of framed image
    img_array = np.array(img)
    # Append dimension at beginning of array (needed for conv2d-input)
    img_array = np.expand_dims(img_array, axis=0)

    # Read in model prediction from image-array
    prediction = int(model.predict(img_array))

    # Make frame grayscale if prediction equals 0
    if prediction == 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Needed for counter
        pred = 0

    # Increment person counter if predicition equals 1
    if prediction == 1 and pred == 0:
        counter += 1
        pred = -1

    # Show prediction and person counter in frame
    cv2.putText(frame, "Prediction: " + str(prediction), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Human counter: " + str(counter), (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Detecting", frame)
    key = cv2.waitKey(1)
    # Quit capture if key "q" is pressed
    if key == ord('q'):
        break


# Exit capturing
cam.release()
cv2.destroyAllWindows()
