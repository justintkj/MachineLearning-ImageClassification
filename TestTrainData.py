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
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn import svm
from sklearn import preprocessing
from deskew import deskew
list_feature = []
list_label = []
for filename in os.listdir('BabyBibs'):
    if filename.endswith(".jpg"): 
        im = Image.open(os.path.join('BabyBibs/', filename)) #relative path to file
        hog = cv2.HOGDescriptor()
        img = cv2.imread(os.path.join('BabyBibs/', filename),0)
        img = deskew(img)
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
        img = deskew(img)
        img = cv2.resize(img,(64,128))
        h = hog.compute(img)
        h = h.ravel()
        list_feature.append(h)
        list_label.append(1)
rndforest = svm.SVC()
rndforest.fit(list_feature,list_label)
#kfold
kf = KFold(n_splits = 4)
kf.get_n_splits(list_feature)
list_feature = np.asarray(list_feature)
list_label = np.asarray(list_label)
count = 0
final_accuracy = 0
for train_index, test_index in kf.split(list_feature):
        X_train, X_test = list_feature[train_index], list_feature[test_index]
        y_train, y_test = list_label[train_index], list_label[test_index]
        rndforest.fit(X_train, y_train)
        y_pred = rndforest.predict(X_test)
        #compute the probability of success given (pred, and correct)
        pred_val = metrics.accuracy_score(y_test, y_pred)
        #Sum the prediction up, to find average later.
        final_accuracy = final_accuracy + pred_val
        count = count +1

print (final_accuracy/count)
