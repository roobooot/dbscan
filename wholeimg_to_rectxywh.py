# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:15:36 2018
现在在这张蚊子上测试没有问题，输入为这张蚊子的图片，
需要的输出在rectlist这个变量上，顺序是    尾巴  头  尾巴  头  尾巴  头  尾巴 头  ……
四个数分别为xmax,xmin,ymax,ymin,稍微调了一下robustness，
有些身体中间没有断开的直接返回原始框，
唯一需要保证的一个条件就是拍摄蚊子时照相机的高度，
因为我断开脚的时候有一个阈值，不然就要调整代码第二十五行核的参数
@author: acer
"""

# -*- coding: utf-8 -*-
from __future__ import division
import cv2
import numpy as np
from PIL import Image

# input picture and save
imagename = "19_0313_9"
img = cv2.imread(imagename + '.jpg', cv2.IMREAD_GRAYSCALE)

# grayprocess, erode using kernel
retval, im_at_fixed = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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


def connectedtobeak(matrix, classes, eroded):
    number = np.zeros(shape=[classes, 1])
    RemoveIndex = {}
    m = matrix.shape[0]
    n = matrix.shape[1]
    for i in range(0, m - 1):
        for j in range(0, n - 1):
            if matrix[i][j] != 0:
                index = matrix[i][j] - 1
                number[index] += 1
                medium = np.array([i, j])
                RemoveIndex.setdefault(str(index), []).append(medium)
    maxnum = 0
    maxnum0 = 0
    ff = 0
    ss = 0
    rect0 = []
    rect1 = []
    for i in range(0, classes):
        if maxnum0 < int(number[i]):
            maxnum0 = number[i]
            ff = i
    number[ff] = 0
    for i in range(0, classes - 1):
        if maxnum < int(number[i]):
            maxnum = number[i]
            ss = i
    if classes != 1:
        xdata0 = []
        ydata0 = []
        for j in range(0, len(RemoveIndex[str(ff)])):
            xdata0.append(RemoveIndex[str(ff)][j][0])
            ydata0.append(RemoveIndex[str(ff)][j][1])
        xmax0 = max(xdata0)
        xmin0 = min(xdata0)
        ymax0 = max(ydata0)
        ymin0 = min(ydata0)
        rect0.append([xmin0, ymin0, xmax0, ymax0])
        xdata1 = []
        ydata1 = []
        for j in range(0, len(RemoveIndex[str(ss)])):
            xdata1.append(RemoveIndex[str(ss)][j][0])
            ydata1.append(RemoveIndex[str(ss)][j][1])
        xmax1 = max(xdata1)
        xmin1 = min(xdata1)
        ymax1 = max(ydata1)
        ymin1 = min(ydata1)
        rect1.append([xmin1, ymin1, xmax1, ymax1])
    return number, RemoveIndex, eroded, rect0, rect1


# main function
output = cv2.connectedComponentsWithStats(erodedpadding, 4, cv2.CV_32S)
num, index, mosquitoesnum, afternoise, rect = getRemovePoints(output[1], output[0], erodedpadding)

retval1, afternoise = cv2.threshold(afternoise, 10, 255, cv2.THRESH_BINARY)
rectlist = []
# save image
for i in range(0, mosquitoesnum):
    # image crop
    cropimage = afternoise[rect[i][0]:rect[i][2], rect[i][1]:rect[i][3]]

    # erode to one point
    kerneleroded = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cropimageeroded = cv2.erode(cropimage, kerneleroded)
    num_eroded = 1
    aheadcropimageeroded = cropimageeroded
    m = cropimageeroded.shape[0]
    n = cropimageeroded.shape[1]
    for j in range(0, m):
        for k in range(0, n):
            if cropimageeroded[j][k] != 0:
                aheadcropimageeroded = cropimageeroded
                cropimageeroded = cv2.erode(cropimageeroded, kerneleroded)
                num_eroded += 1

    # image dilation using the ratio of WH
    ratio = (rect[i][2] - rect[i][0] - 4) / (rect[i][3] - rect[i][1] - 4)
    if ratio < 1:
        kerneleroded1_x = int((2 * num_eroded + 1) * (rect[i][2] - rect[i][0] - 4) / (rect[i][3] - rect[i][1] - 4))
        kerneleroded1_y = int(2 * num_eroded + 1)
    else:
        kerneleroded1_y = int((2 * num_eroded + 1) * (rect[i][3] - rect[i][1] - 4) / (rect[i][2] - rect[i][0] - 4))
        kerneleroded1_x = 2 * num_eroded + 1
    kerneleroded1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kerneleroded1_x, kerneleroded1_y))
    cropimageerodeddilated = cv2.dilate(aheadcropimageeroded, kerneleroded1)

    # difference
    image_diffe = cropimage - cropimageerodeddilated
    retval1, image_diffe = cv2.threshold(image_diffe, 10, 255, cv2.THRESH_BINARY)
    output1 = cv2.connectedComponentsWithStats(image_diffe, 4, cv2.CV_32S)
    if output1[0] != 1:
        num1, index1, afternoise1, rect0, rect1 = connectedtobeak(output1[1], output1[0], image_diffe)
        rect0a = []
        rect1a = []

        cropimage0 = cropimage[rect0[0][0]:rect0[0][2], rect0[0][1]:rect0[0][3]]
        cropimage1 = cropimage[rect1[0][0]:rect1[0][2], rect1[0][1]:rect1[0][3]]

        b = rect[i][1] + rect0[0][0]
        rect0a.append(b)
        b = rect[i][1] + rect0[0][1]
        rect0a.append(b)
        b = rect[i][3] + rect0[0][2]
        rect0a.append(b)
        b = rect[i][3] + rect0[0][3]
        rect0a.append(b)
        rectlist.append(rect0a)

        b = rect[i][1] + rect1[0][0]
        rect1a.append(b)
        b = rect[i][1] + rect1[0][1]
        rect1a.append(b)
        b = rect[i][3] + rect1[0][2]
        rect1a.append(b)
        b = rect[i][3] + rect1[0][3]
        rect1a.append(b)
        rectlist.append(rect1a)

        cropimage3 = Image.fromarray(cropimage0)
        cropimage3.save('edcurrent\\cropimage' + str(i) + '_onlytail.jpg')
        cropimage3.save('edhistory\\' + imagename + 'cropimage' + str(i) + '_onlytail.jpg')
        cropimage4 = Image.fromarray(cropimage1)
        cropimage4.save('edcurrent\\cropimage' + str(i) + '_onlyhead.jpg')
        cropimage4.save('edhistory\\' + imagename + 'cropimage' + str(i) + '_onlyhead.jpg')
