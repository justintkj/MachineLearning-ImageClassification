import cv2
import numpy as np
import os
import sklearn
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import metrics
from scipy import stats
import timeit
from ConfusionMatrix import confusionMatrixAlgo
from sklearn.externals import joblib
from sklearn.svm import NuSVC
from sklearn import grid_search
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from sklearn.decomposition import PCA
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA

pca = IncrementalPCA(n_components = 500)
def get_feature_space(images):
    features = []
    count = 1
    buffer = []
    for image in images:
        des = hog(image)
        features.extend(des.reshape(-1,36).tolist())
        if(count %100 ==0):
            print str(count) + " out of " + str (len(images))
        count+= 1
    return features
def to_BOW_features(features,codebook):
    BOW = [codebook.predict(feature) for feature in features]
    hist = [np.histogram(bag, bins = codebook.n_clusters)[0]for bag in BOW]
    return hist
#def to_hog_features(images):
#    return [hog(r).reshape(-1,36).tolist() for r in images]
def to_hog_features(images):
    return [hog(images).reshape(-1,36).tolist()]



path = "Pictures\\"
#start_time = timeit.default_timer()
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
count = 0
print "one"
for x in train_name_path:
    im = cv2.imread(x, 0)
    if im is not None:
        #hog = cv2.HOGDescriptor()
        #hog.winSize = (32,64)
        img = cv2.resize(im,(32,64),interpolation = cv2.INTER_AREA)
        if(count %100 ==0):
            print str(count) + " out of " + str (len(train_name_path))
        count+= 1
        if img is not None:
            descriptor.extend(to_hog_features(img))
            for y in range (0,18):
                if(train_names[y] in x):
                    label.append(y)
print ("two")
#print descriptor
#feature_space_hog = get_feature_space(descriptor)
#X_train_hog = to_hog_features(descriptor)
km = MiniBatchKMeans(100)
buffer = []
index = 0
for mini in descriptor:
    buffer.extend(mini)
    index += 1
    if(index %101 == 0):
        km.partial_fit(buffer)
        buffer = []
    if(index %1000 == 0):
        print (str(index) + "out of "+ str(len(descriptor)))
#km = MiniBatchKMeans(25, n_jobs = 1).fit(feature_space_hog)
#km = MiniBatchKMeans(2000,batch_size = 7500,verbose = 1).fit(feature_space_hog)
X_train_bow = to_BOW_features(descriptor,km)
normalizer = preprocessing.Normalizer().fit(np.array(X_train_bow))
X_train_bow = normalizer.transform(np.array(X_train_bow))
#svc = RandomForestClassifier(oob_score = True)
svc = SVC()
param_dist = {}
param_dist['C'] = stats.uniform(7, 9)
param_dist['kernel'] = ['rbf']
#param_dist['gamma'] = stats.uniform(0.1, 0.1)
#param_dist['degree'] = stats.randint(2, 20)
#param_dist['C'] = [2,4,6,8]
#param_dist['kernel'] = ['rbf']
#param_dist['gamma'] = [0.19]
#param_dist['degree'] = [3]
random_search = RandomizedSearchCV(svc, param_distributions = param_dist, n_iter = 2)
random_search.fit(X_train_bow, label)
#svc.fit(X_train_bow, label)
joblib.dump(random_search, 'model_svm4.pkl', protocol=2) #Save Model
joblib.dump(km, 'km4.pkl', protocol=2)

#print svc.oob_score_
print random_search.best_params_
print random_search.best_score_
