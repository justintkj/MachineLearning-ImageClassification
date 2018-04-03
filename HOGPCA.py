import cv2
import numpy as np
import os
import sklearn
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics
import timeit
from ConfusionMatrix import confusionMatrixAlgo
from sklearn.externals import joblib
from sklearn.svm import NuSVC
from sklearn import grid_search
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier

pca = PCA(n_components = 50)

path = "Pictures\\"
start_time = timeit.default_timer()
train_folder = os.listdir(path)
train_names = []
train_name_path = []
for x in train_folder:
    train_subfolder = os.listdir("Pictures\\" + x)
    train_names.append(x)
    for y in train_subfolder:
        train_name_path.append("Pictures\\" + x + "\\" + y)

descriptor = []
label =[]
i=0
print "one"
for x in train_name_path:
    im = cv2.imread(x, 0)
    if im is not None:
        #hog = cv2.HOGDescriptor()
        #hog.winSize = (32,64)
        img = cv2.resize(im,(16,32))
        h = hog(img, orientations=8, pixels_per_cell=(4, 4),
               cells_per_block=(4, 4))
        #print ("two")
        #h = hog.compute(img)
        #print ("two")
        #h = [item[0] for item in h]
        if h is not None:
            descriptor.append(h)
            for y in range (0,18):
                if(train_names[y] in x):
                    label.append(y)
print ("two")

n = np.array(descriptor)
descriptor = pca.fit_transform(n)
normalizer = preprocessing.Normalizer().fit(descriptor)
descriptor = normalizer.transform(descriptor)
#svm = RandomForestClassifier()
#svm.fit(n, np.array(label))
#svm = SVC(C=4, gamma= 0.4000000000000001)
#svm.fit(descriptor, np.array(label))
#svm = LinearSVC(random_state=0)
#svm.fit(np.array(descriptor),np.array(label))
svc = SVC()
param_grid = [
  {'C': [1,10,100,1000,10000], 'kernel': ['linear']},
  {'C': [1,10, 100, 1000,10000],'gamma': [0.1,0.01,0.001, 0.0001], 'kernel': ['rbf']},
 ]
grid_search = GridSearchCV(svc, param_grid)
grid_search.fit(np.array(descriptor), np.array(label))
svm = grid_search
print ("three")
joblib.dump(normalizer, 'normalizer2.pkl', protocol=2)
joblib.dump(svm, 'model_svm2.pkl', protocol=2) #Save Model
joblib.dump(pca, 'pca2.pkl', protocol=2)
confusion = np.zeros((18,18))
count = 0
matrix = []
for x in train_name_path:
    im = cv2.imread(x,0)
    if im is not None:
        #hog.winSize = Size(16,32)
        img = cv2.resize(im,(16,32))
        h = hog(img, orientations=8, pixels_per_cell=(4, 4),
           cells_per_block=(4, 4))
        if h is not None:
            h = np.reshape(tempy, (1,-1))
            h = pca.transform(h)
            tempy = normalizer.transform(h)
            p = svm.predict(tempy)
            matrix.append(p)
            confusion[label[count], p] = confusion[label[count],p]+1
            count = count + 1
#con_matrix = confusionMatrixAlgo(label, matrix)
#confusion = confusion/(confusion.sum(axis=1))

#confusion = confusion.transpose()
num_correct = 0
total_num = 0
for x in range (len(confusion)) :
    for y in range (len(confusion[0])):
        if x == y:
            num_correct = num_correct + confusion[x][y]
        total_num = total_num + confusion[x][y]
print "percentage"
print num_correct/ total_num
total_time = (timeit.default_timer() - start_time)*1000

print confusion
print total_time
print svm.best_params_
print svm.best_score_
