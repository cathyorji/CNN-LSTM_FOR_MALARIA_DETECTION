import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
best_model_file = "/content/drive/MyDrive/model_10.keras"
model = tf.keras.models.load_model(best_model_file)
print(model.summary())

input_shape = (50, 50)
categories = ["infected" , "uninfected"]
def prepareImage(img):
    resized = cv2.resize(img , input_shape, interpolation=cv2.INTER_AREA)
    imgResult = np.expand_dims(resized , axis=0)
    imgResult =   imgResult / 255.
    return imgResult
  
# load test image
testImagePath = "/content/drive/MyDrive/Uninfected/C100P61ThinF_IMG_20150918_144104_cell_128.png"
img = cv2.imread(testImagePath)

# prepare image for the model
imgForModel = prepareImage(img)

# run the prediction
result = model.predict(imgForModel, verbose=1)
print(result)

# binary classification
if result >0.5 :
    result = 1
else :
    result = 0

print(result)
text = categories[result]
