from keras.models import load_model
import numpy as np
import cv2
import matplotlib as plt

model = load_model('model3.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

img = cv2.imread('../dataset/valid/2/12_100.jpg')

i = np.expand_dims(img, axis=0)

print(model.predict_classes(i))

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

