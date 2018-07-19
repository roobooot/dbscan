import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from PIL import Image
#print(__doc__)
def From2D2points(img):
    points = list()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] != 0:
                points.append([i, j])
    points = np.array(points)
    return points
def num_pointInlable(labels):

    classes = list(set(labels))
    num_pointInlable = dict()
    for i in range(0, len(labels)):
        if labels[i] in classes:
            num_pointInlable.setdefault(str(labels[i]), []).append(i)
    return num_pointInlable

imgname = 'crop1.jpg'
img = cv2.imread(imgname, cv2.IMREAD_UNCHANGED)
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, imgbin = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
imgbin = img
points = From2D2points(imgbin)
X = points
X1 = StandardScaler().fit_transform(X)
db = DBSCAN(eps=0.03, min_samples=3).fit(X1)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels))
print('after 1st clustering, the num of all clusterings: ',n_clusters_)
num_pointInlable = num_pointInlable(labels)
##Covariance
X2Cov = list()
Y2Cov = list()
for a in num_pointInlable[str(-1)]:
    X2Cov.append(X[a][0])
    Y2Cov.append(X[a][1])
X2Cov = np.array(X2Cov)
Y2Cov = np.array(Y2Cov)
NoisyCov = np.cov(X2Cov,Y2Cov)
# delete noisy
print('length of X before delete: ', len(X))
for i in set(labels):
    if len(num_pointInlable[str(i)])<1000:
        print(i, ',deleting points: ', len(num_pointInlable[str(i)]))
        for a in num_pointInlable[str(i)]:
            imgbin[X[a][0]][X[a][1]] = 0#delete noisy
X2 = From2D2points(imgbin)
X2 = StandardScaler().fit_transform(X2)
db = DBSCAN(eps=0.05, min_samples=200).fit(X2)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('after 2nd clustering, the num of mosq: ', n_clusters_)
# Black removed and is used for noise instead.
unique_labels = set(labels)

img = Image.fromarray(imgbin)
img.show()


