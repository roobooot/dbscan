# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
from PIL import Image

# input picture and save
imagename = "19_0313_9"
img = cv2.imread(imagename + '.jpg', cv2.IMREAD_GRAYSCALE)

# grayprocess, erode using kernel
retval, im_at_fixed = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
eroded = cv2.erode(im_at_fixed, kernel)
# padding 0
erodedpadding = cv2.copyMakeBorder(eroded, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)


# another connected components function:ret, markers = cv2.connectedComponents(eroded,)
# remove the noise, input data form: the output matrix of cv2.connectedComponentsWithStats,
# number of classes, input image
def getRemovePoints(matrix, classes, eroded):
    number = np.zeros(shape=[classes, 1])
    RemoveIndex = {}
    m = matrix.shape[0]
    n = matrix.shape[1]
    rect = []
    b = 0
    for i in range(0, m - 1):
        for j in range(0, n - 1):
            if matrix[i][j] != 0:
                index = matrix[i][j] - 1
                number[index] += 1
                medium = np.array([i, j])
                RemoveIndex.setdefault(str(index), []).append(medium)
    for i in range(0, classes - 1):
        if number[i] < 1000:
            for j in range(0, len(RemoveIndex[str(i)])):
                eroded[RemoveIndex[str(i)][j][0]][RemoveIndex[str(i)][j][1]] = 0  # removenoise
            del RemoveIndex[str(i)]
        else:
            xdata = []
            ydata = []
            for j in range(0, len(RemoveIndex[str(i)])):
                xdata.append(RemoveIndex[str(i)][j][0])
                ydata.append(RemoveIndex[str(i)][j][1])
            xmax = max(xdata) + 2
            xmin = min(xdata) - 2
            ymax = max(ydata) + 2
            ymin = min(ydata) - 2
            b += 1
            # height=xmax-xmin+1
            # width=ymax-ymin+1
            rect.append([xmin, ymin, xmax, ymax])
    return number, RemoveIndex, b, eroded, rect


# main function
output = cv2.connectedComponentsWithStats(erodedpadding, 4, cv2.CV_32S)
num, index, mosquitoesnum, afternoise, rect = getRemovePoints(output[1], output[0], erodedpadding)
# num: n*2 array; index: dict{'1':[x,y]...} mosquitoesnum  ; afternoise: img after removal noisy; rect: array of bbp.
# save image
for i in range(0, mosquitoesnum):
    cropimage = afternoise[rect[i][0]:rect[i][2], rect[i][1]:rect[i][3]]
    cropimage1 = Image.fromarray(cropimage)
    cropimage1.save('current\\cropimage' + str(i) + '.jpg')
    cropimage1.save('history\\' + imagename + 'cropimage' + str(i) + '.jpg')

'''
img1.save('image5.jpg')
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.imwrite('image',eroded)
cv2.imshow('image',img)
cv2.waitKey (0)
'''
