import numpy as np
import cv2 as cv
import keras.applications.xception as xception
import keras
import tensorflow as tf
from keras.layers import Lambda
categories={0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard', 4: 'clothes', 5: 'green-glass', 6: 'metal', 7: 'paper', 8: 'plastic', 9: 'shoes', 10: 'trash', 11: 'white-glass'}
model=keras.models.load_model('model.h5')
image_path = "garbage_classification/shoes/shoes1.jpg"
img = cv.imread(image_path)
image = cv.resize(img, (320, 320))

# Add batch dimension to the image
image = np.expand_dims(image, axis=0)

# Prediction
print("Image shape:", image.shape)
pred_class = model.predict(image)
predicted_category = categories[np.argmax(pred_class)]
print(f"Predicted category: {predicted_category}")