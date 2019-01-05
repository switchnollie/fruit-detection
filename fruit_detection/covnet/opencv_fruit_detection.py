#!/bin/python3.6

import cv2
import numpy as np
from PIL import Image
from keras import models

# Image size (one axis)
size = 100
# Initialize person counter
counter = pred = 0
# Init label
label = ""

# Load saved cnn-model
model = models.load_model('model3.h5')

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
    prediction = int(model.predict_classes(img_array))
    # Print label instead of prediction
    if prediction == 1:
        label = "Apple"
    elif prediction == 2:
        label = "Banana"
    elif prediction == 3:
        label = "Pear"
    elif prediction == 4:
        label = "Orange"

    # Show prediction and person counter in frame
    cv2.putText(frame, "Prediction: " + label, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show frame
    cv2.imshow("Detecting", frame)
    key = cv2.waitKey(1)
    # Quit capture if key "q" is pressed
    if key == ord('q'):
        break


# Exit capturing
cam.release()
cv2.destroyAllWindows()
