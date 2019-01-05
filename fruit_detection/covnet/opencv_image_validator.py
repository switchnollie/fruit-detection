#!/bin/python3.6

import numpy as np
import cv2
from PIL import Image
from keras import models

# Image size (one axis)
size = 100
# Init label
label = ""

# Load saved cnn-model
model = models.load_model('model3.h5')

# Capture onboard webcam
cam = cv2.imread("./fruits_dataset/valid/pear/1.jpg")

# ... in RGB-mode
img = Image.fromarray(cam, 'RGB')
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

print(prediction, label)

while True:

    cv2.imshow("Image", cam)

    key = cv2.waitKey(1)
    # Quit capture if key "q" is pressed
    if key == ord('q'):
        break
