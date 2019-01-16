#!/bin/python3.6

import sys
import os
import cv2
from PIL import Image

# Command line arguments (path, number of images)
path = sys.argv[1]
img_count = int(sys.argv[2])

if len(sys.argv) != 3:
    print("Error! Two arguments needed!")
    exit()

# Makes all intermediate-level directories needed to contain the leaf path
os.makedirs(path, exist_ok=True)

# Initialize filename (is counter too)
filename = 0

# Capture onboard webcam
cam = cv2.VideoCapture(0)

# Record images
while True and filename < img_count:

    # Increment filename and counter
    filename += 1

    # Read in webcam frame...
    _, frame = cam.read()

    # ... in RGB-mode
    img = Image.fromarray(frame, 'RGB')
    # Resize img (needed for CNN especially inputs)
    img = img.resize((128, 128))
    # Save image to path as "JPEG" by joining path with filename and dataformat
    img.save(os.path.join(path, str(filename) + ".jpg"), "JPEG")

    # Show captured frame
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    # Quit capture if key "q" is pressed
    if key == ord('q'):
        break


# Exit capturing
cam.release()
cv2.destroyAllWindows()
