import numpy as np
import cv2 as cv
import keras.applications.xception as xception
import keras
import tensorflow as tf
from keras.layers import Lambda
from keras.layers import Lambda
categories={0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard', 4: 'clothes', 5: 'green-glass', 6: 'metal', 7: 'paper', 8: 'plastic', 9: 'shoes', 10: 'trash', 11: 'white-glass'}
model=keras.models.load_model('model.h5')
cap=cv.VideoCapture(0)
while True:
    ret,frame=cap.read()
    frame=cv.resize(frame,(320,320))
    frame=cv.flip(frame,1)
    image = np.expand_dims(frame, axis=0)
    pred = model.predict(image)
    cv.putText(frame,categories[np.argmax(pred)],(30,40),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
    if not ret:
        break
    cv.imshow("Imagen",frame)
    key=cv.waitKey(1)
    if key==27: #esc
        break
cap.release()
cv.destroyAllWindows()