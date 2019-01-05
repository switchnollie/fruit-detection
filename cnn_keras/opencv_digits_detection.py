#!/bin/python3.6

import cv2
import numpy as np
from keras import models

size = 28
label = ""

model = models.load_model('cnn.h5')

cam = cv2.VideoCapture(0)

while True:

    _, frame = cam.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img.resize((size, size))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=3)

    prediction = model.predict_classes(img_array)
    cv2.putText(frame, "Prediction: " + str(prediction), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Detecting", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()
