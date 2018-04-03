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
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

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
#Create Deskriptor
orb = cv2.xfeatures2d.SIFT_create(nfeatures = 10)
print("one")
#Bag Of Words
BOW = cv2.BOWKMeansTrainer(18)

for x in train_name_path:
    #print ("one")
    img = cv2.imread(x,0)
    #img = img.astype(np.uint8, copy=False)
    if img is not None:
        img = cv2.resize(img, (64,128))
        kp, des = orb.detectAndCompute(img, None)
        if des is not None:
            #des = np.float32(des)
            BOW.add(des)

dic = BOW.cluster()
print ("two")
index_parameter = dict(algorithm = 0, trees =18)
search_parameter = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_parameter, search_parameter)
orb2 = cv2.xfeatures2d.SIFT_create(nfeatures = 10)
#bowDiction = cv2.BOWImgDescriptorExtractor(orb2, cv2.BFMatcher(cv2.NORM_L2))
bowDiction = cv2.BOWImgDescriptorExtractor(orb2, flann)
bowDiction.setVocabulary(dic)
print "bow dictionary", np.shape(dic)

descriptor = []
label =[]
i=0
for x in train_name_path:
    im = cv2.imread(x, 0)
    if im is not None:
        hog = cv2.HOGDescriptor()
        img = cv2.resize(im,(64,128))
        h = hog.compute(img)
        h = h.ravel()
        h = h[::10]
        #h = h[:int(len(h)*0.1)]
        temp = bowDiction.compute(img,orb.detect(img))
        tempy = []
        if temp is not None:
            tempy.extend(temp[0])
            tempy.extend(h)
            #temp = np.vstack([bowDiction.compute(im,orb.detect(im)), h])
            descriptor.append(tempy)
            for y in range (0,18):
                if(train_names[y] in x):
                    label.append(y)
#svm = SVC(C = 4, gamma = 0.4)
svm = RandomForestClassifier(oob_score = True)
#svm.fit(np.array(descriptor), np.array(label))
#svm = LinearSVC(random_state=0)
#svm.fit(np.array(descriptor),np.array(label))
#svc = SVC()
#param_grid = [
#  {'C': [3.5,4,4.5],'gamma':[0.35,0.4]}]
#grid_search = GridSearchCV(svm, param_grid = param_grid)
#grid_search.fit(np.array(descriptor), np.array(label))
#svm = grid_search
svm = svm.fit(np.array(descriptor), np.array(label))
print(svm.oob_score_)
print ("three")
joblib.dump(orb, 'model_sift.pkl',protocol=2)
joblib.dump(svm, 'model_svm.pkl', protocol=2) #Save Model
joblib.dump(dic, 'bow_dic.pkl', protocol=2)
confusion = np.zeros((18,18))
count = 0
matrix = []
for x in train_name_path:
    im = cv2.imread(x,0)
    if im is not None:
        hog = cv2.HOGDescriptor()
        #hog.winSize = Size(16,32)
        img = cv2.resize(im,(64,128))
        temp = bowDiction.compute(img,orb.detect(img))
        tempy = []
        h = hog.compute(img)
        h = h.ravel()
        h = h[::10]
        #h = h[:int(len(h)*0.1)]
        if temp is not None:
            tempy.extend(temp[0])
            tempy.extend(h)
            tempy = np.reshape(tempy,(1,-1))
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
