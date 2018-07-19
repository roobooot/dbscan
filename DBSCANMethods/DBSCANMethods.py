import numpy as np


def From2D2points(img):
    points = list()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] != 0:
                points.append([i, j])
    points = np.array(points)
    return points


def num_pointInlable(self, labels):
    # {'-1':[1,2,34,56,4]'0':[234,232,521,231]}
    classes = list(set(labels))
    num_pointInlable = dict()
    for i in range(0, len(labels)):
        if labels[i] in classes:
            num_pointInlable.setdefault(str(labels[i]), []).append(i)
    return num_pointInlable
