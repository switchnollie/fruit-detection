#!/bin/python3.6

import cv2
from keras import models
import numpy as np
from PIL import Image


# Load the saved model
model = models.load_model('cnn.h5')
# Load webcam feed
cap = cv2.VideoCapture(0)

while True:
    # Read webcam feed
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(gray_frame)
    # Resize model to size we trained the model
    img = img.resize((28, 28))
    img_array = np.array(img)
    # img_array = np.expand_dims(img_array, -1)
    img_array = np.expand_dims(img_array, 0)
    img_array = np.expand_dims(img_array, 3)
    # Call predict method on model to predict image
    prediction = int(model.predict(img_array)[0][0])
    if prediction != 0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.putText(frame, "Prediction : " + str(prediction), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Show image frame
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
