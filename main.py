#auth : janaki

import os
import sys
import datetime
import glob as glob
import numpy as np
import cv2
import keras
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.optimizers import SGD
import tensorflow
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# global parameters
width, height = 224, 224
epochs = 2
batchSize = 32
trainDir = 'train'
classifClasses = len(glob.glob(trainDir + '/*'))

# read input data
samples = 0
for r, dirs, files in os.walk(trainDir):
    for dir in dirs:
        samples += len(glob.glob(os.path.join(r, dir + "/*")))

# preprocess input data
trainDatagen =  ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    fill_mode = 'nearest',
    horizontal_flip = True)
# load and preprocess test data
testData = glob.glob("testImages/*.jpg")
image = cv2.imread(testData[0])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256)).astype(np.float32)
image = np.expand_dims(image, axis = 0)/255

	
# generate training object from train data directory
trainGenerator = trainDatagen.flow_from_directory(
    trainDir,
    target_size = (width, height),
    batch_size = batchSize)
	
# model
model = VGG16(weights = 'imagenet', include_top = False)
X = model.output
X = GlobalAveragePooling2D()(X)
X = Dense(1024, activation='relu')(X)
predicted = Dense(classifClasses, activation = 'softmax')(X)
finalModel = Model(inputs = model.input, outputs = predicted)
for layer in model.layers:
    layer.trainable = False

# optimize the model
finalModel.compile(optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])
	
# fit model
model_history = finalModel.fit_generator(
    trainGenerator,
    epochs = epochs,
    steps_per_epoch = samples)
	
# predict test data classes
outPred = []
imageId = []
for rec in testData:
    image = cv2.imread(rec)
    image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32) / 255.0
    image = np.expand_dims(image, axis =0)
    y = [np.argmax(model.predict(image))]
    outPred.extend(list(y))
    imageId.extend([rec.rsplit("/")[-1]])	
output = pd.DataFrame()
output["id"] = imageId
output["category"] = outPred

# Accuracy calculation
finalModel_history.history['acc']
y_test = pd.read_csv('test.csv')
correctlyClassified = 0
for ind in output.index:
  for ind1 in y_test.index:
    if (output['id'][ind] == y_test['id'][ind1]) and (output['category'][ind] == y_test['category'][ind1]):
      correctlyClassified = correctlyClassified + 1
      break
accuracy = (correctlyClassified / len(output))*100
print("Test Accuracy : ", accuracy)
