# -*- coding: utf-8 -*-
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_sample_image
from sklearn.utils import Bunch
import os
import cv2
def load_sample_images():
    """Load sample images for image manipulation.

    Loads both, ``china`` and ``flower``.

    Returns
    -------
    data : Bunch
        Dictionary-like object with the following attributes : 'images', the
        two sample images, 'filenames', the file names for the images, and
        'DESCR' the full description of the dataset.

    Examples
    --------
    To load the data and visualize the images:

    >>> from sklearn.datasets import load_sample_images
    >>> dataset = load_sample_images()     #doctest: +SKIP
    >>> len(dataset.images)                #doctest: +SKIP
    2
    >>> first_img_data = dataset.images[0] #doctest: +SKIP
    >>> first_img_data.shape               #doctest: +SKIP
    (427, 640, 3)
    >>> first_img_data.dtype               #doctest: +SKIP
    dtype('uint8')
    """
    # Try to import imread from scipy. We do this lazily here to prevent
    # this module from depending on PIL.
    try:
        try:
            from scipy.misc import imread
        except ImportError:
            from scipy.misc.pilutil import imread
    except ImportError:
        raise ImportError("The Python Imaging Library (PIL) "
                          "is required to load data from jpeg files")
    module_path = os.path.join(ROOT_Dir, "images")
    with open(os.path.join(module_path, 'README.txt')) as f:
        descr = f.read()
    filenames = [os.path.join(module_path, filename)
                 for filename in os.listdir(module_path)
                 if filename.endswith(".jpg")]
    # Load image data for each image in the source folder.
    images = [imread(filename) for filename in filenames]

    return Bunch(images=images,
                 filenames=filenames,
                 DESCR=descr)


def load_sample_image(image_name):
    """Load the numpy array of a single sample image

    Parameters
    -----------
    image_name : {`china.jpg`, `flower.jpg`}
        The name of the sample image loaded

    Returns
    -------
    img : 3D array
        The image as a numpy array: height x width x color

    Examples
    ---------

    >>> from sklearn.datasets import load_sample_image
    >>> china = load_sample_image('china.jpg')   # doctest: +SKIP
    >>> china.dtype                              # doctest: +SKIP
    dtype('uint8')
    >>> china.shape                              # doctest: +SKIP
    (427, 640, 3)
    >>> flower = load_sample_image('flower.jpg') # doctest: +SKIP
    >>> flower.dtype                             # doctest: +SKIP
    dtype('uint8')
    >>> flower.shape                             # doctest: +SKIP
    (427, 640, 3)
    """
    images = load_sample_images()
    index = None
    for i, filename in enumerate(images.filenames):
        if filename.endswith(image_name):
            index = i
            break
    if index is None:
        raise AttributeError("Cannot find sample image: %s" % image_name)
    return images.images[index]

# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                            random_state=0)
ROOT_Dir = os.getcwd()
img_Dir = os.path.join(ROOT_Dir, 'images', 'mosquito.jpg')
imgname1 = 'mosquito.jpg'
imgname2 = 'crop_mos.jpg'
imgname3 = 'mos2.jpg'
img = cv2.imread(imgname3)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval, im_at_fixed = cv2.threshold(img_grey, 150, 255, cv2.THRESH_BINARY)
#cv2.imshow('image', im_at_fixed)
#im_at_fixed = cv2.imread(imgname3)
points = list()
for i in range(0,im_at_fixed.shape[0]):
    for j in range(0, im_at_fixed.shape[1]):
        if im_at_fixed[i][j] == 0:
            points.append([i,j])
points = np.array(points)
X = points
#X = StandardScaler().fit_transform(X)

#ROOT_Dir = os.getcwd()
#image_Name = 'mosquito.jpg'
#image_Dir = os.path.join(ROOT_Dir,image_Name)
#mosquito = load_sample_image(image_Name)
#mosquito = np.array(mosquito, dtype=np.float64)/225
#w, h, d = original_shape = tuple(mosquito.shape)
#assert d == 3
#image_array = np.reshape(mosquito, (w * h, d))
# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=1, min_samples=11).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
'''
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))
'''
# #############################################################################
# Plot result
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=20)
    plt.plot(600,600,'.',markersize=20)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=3)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

