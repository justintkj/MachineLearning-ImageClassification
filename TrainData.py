from __future__ import with_statement
from PIL import Image
import os
import numpy as np
from skimage.feature import hog
from skimage import data, exposure
import cv2
import base64
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

list_feature = []
list_label = []
for filename in os.listdir('BabyBibs'):
    if filename.endswith(".jpg"): 
        im = Image.open(os.path.join('BabyBibs/', filename)) #relative path to file
        hog = cv2.HOGDescriptor()
        img = cv2.imread(os.path.join('BabyBibs/', filename),0)
        img = cv2.resize(img,(64,128))
        h = hog.compute(img)
        h = h.ravel()
        list_feature.append(h)
        list_label.append(0)
for filename in os.listdir('BabyHat'):
    if filename.endswith(".jpg"): 
        im = Image.open(os.path.join('BabyHat/', filename)) #relative path to file
        hog = cv2.HOGDescriptor()
        img = cv2.imread(os.path.join('BabyHat/', filename),0)
        img = cv2.resize(img,(64,128))
        h = hog.compute(img)
        h = h.ravel()
        list_feature.append(h)
        list_label.append(1)
rndforest = RandomForestClassifier(random_state=1)
rndforest.fit(list_feature,list_label)

#sample data
im = Image.open('Test_9.jpg')
hog = cv2.HOGDescriptor()
img = cv2.imread('Test_9.jpg',0)
img = cv2.resize(img,(64,128))
h = hog.compute(img)
h = h.ravel()
test = []
test.append(h)
test.append(h)
print (rndforest.predict(test))
