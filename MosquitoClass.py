# -*-coding:utf-8-*-
"""
This is the whole code for the traditional method to do the detection for mosquitoes

@@author: Zeyu Lu, Guanqun Huang
"""
import cv2
import numpy as np
import os
import glob
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from skimage import draw


##pre
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
        if number[i] < 1000 or number[i]>7000:
            for j in range(0, len(RemoveIndex[str(i)])):
                eroded[RemoveIndex[str(i)][j][0]][RemoveIndex[str(i)][j][1]] = 0  # removenoise
            del RemoveIndex[str(i)]
        else:
            xdata = []
            ydata = []
            for j in range(0, len(RemoveIndex[str(i)])):
                xdata.append(RemoveIndex[str(i)][j][0])
                ydata.append(RemoveIndex[str(i)][j][1])
            xmax = max(xdata) +1
            xmin = min(xdata) -1
            ymax = max(ydata) +1
            ymin = min(ydata) -1
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


# Seperate every solo mosquito and store in a list
def SeperateAndStoreInList(img, mosquitoesnum, rect, imagename, IfStore=False):
    # OUTPUT: soloImage: [
    # [points of 1st mos]
    # [points of 2nd mos]
    # ...
    # ]----type: list
    soloImage = []
    if IfStore:
        folder1 = os.getcwd() + '\\SavaImages\\CropMos'
        # 获取此py文件路径，在此路径选创建在new_folder文件夹中的test文件夹
        folder2 = os.path.join(folder1, 'current')
        folder3 = os.path.join(folder1, 'history')

        if not os.path.exists(folder1):
            os.makedirs(folder1)
        if not os.path.exists(folder2):
            os.makedirs(folder2)
        if not os.path.exists(folder3):
            os.makedirs(folder3)
        filelist = glob.glob(os.path.join(folder2, "*.jpg"))
        for f in filelist:
            os.remove(f)
    for i in range(0, mosquitoesnum):
        cropimage = img[rect[i][0]:rect[i][2], rect[i][1]:rect[i][3]]
        soloImage.append(list(cropimage))
        if IfStore:
            cropimage = Image.fromarray(cropimage)
            cropimage.save(folder2 + '\\cropimage' + str(i) + '.jpg')
            cropimage.save(folder3 + '\\' + imagename + 'cropimage' + str(i) + '.jpg')
    return soloImage


##Clustering
def From2D2points(img):
    # points:[i1, j1]
    #        [i2, j2]
    #        ...
    #
    points = list()
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i][j] > 200:
                points.append([i, j])
    points = np.array(points)
    return points


def gt_num_pointInlable(labels):
    classes = list(set(labels))
    num_pointInlable = dict()
    for i in range(0, len(labels) - 1):
        if labels[i] in classes:
            num_pointInlable.setdefault(str(labels[i]), []).append(i)
    return num_pointInlable


def StoreSeperateImg(soloimgShape, points, pointstoshow, rank, key, path):
    # show the points that you'd like to show and store
    height, width = soloimgShape
    img = np.zeros([height, width])
    PointsX = points[:, 0]
    PointsY = points[:, 1]
    for i in range(0, len(pointstoshow)):
        img[PointsX[pointstoshow[i]]][PointsY[pointstoshow[i]]] = 255
    img = Image.fromarray(img)
    img = img.convert('RGB')
    img.save(path + 'clustering' + str(rank) + '.' + str(key) + '.jpg')


def dbscanFromIMG(img, eps, min_samples):
    # input: img,eps,min_samples
    # output: dbscan, 2Dpoints from img
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgbin = img  ##...
    points = From2D2points(imgbin)
    X = points
    X1 = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X1)
    return db, points


def dbscanAndCovarianceForSoloMos(soloImage, mosquitoesnum):
    # output:
    #    detectedby2: the bad detection by clustering, 2 position were detected out.
    #    ImgAfterClustering: it is a dict, {'0': {'-1': [a2, b2, c2,d2], '0': [a1, b1, c1, d1]},
    #                                       '1': {'-1': [a2, b2, c2,d2], '0': [a1, b1, c1, d1]}
    #                                           ...        }
    #
    dectedby2 = list()
    ImgAfterClustering = dict()
    Storepath = os.path.join(os.getcwd(), 'SavaImages\\afterclustering\\Mosquito\\current\\')
    if not os.path.exists(Storepath):
        os.makedirs(Storepath)
    filelist = glob.glob(os.path.join(Storepath, "*.jpg"))
    for f in filelist:
        os.remove(f)
    IfPrintCovrariance = True
    pointsOfAllMos = dict()  # Store all the points of every solo mos in mos bounding box.
    for i in range(0, len(soloImage)):
        soloimg = soloImage[i]
        soloimg = np.array(soloimg)
        soloimgShape = soloimg.shape
        db, points = dbscanFromIMG(soloimg, 0.5, 200)  # Clustering
        pointsOfAllMos[str(i)] = points  # 每张文字图片里点的坐标
        X = points
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels))
        print('No.', i, '\nafter clustering, the num of all clusterings: ', n_clusters_)
        num_pointInlable = gt_num_pointInlable(labels)
        ImgAfterClustering[str(i)] = num_pointInlable
        for key in num_pointInlable.keys():
            print('cluster: ', key, 'num of points:', len(num_pointInlable[key]))
            StoreSeperateImg(soloimgShape, points, np.array(num_pointInlable[key]), i, key, Storepath)
        ##Covariance
        if '-1' in num_pointInlable.keys():
            X2Cov = list()
            Y2Cov = list()
            for a in num_pointInlable[str(-1)]:
                X2Cov.append(X[a][0])
                Y2Cov.append(X[a][1])
            X2Cov = np.array(X2Cov)
            Y2Cov = np.array(Y2Cov)
            NoisyCov = np.cov(X2Cov, Y2Cov)
            if IfPrintCovrariance:
                print('No.', i, ',Covariance:\n', NoisyCov)
            if NoisyCov[0][0] > 170 or NoisyCov[1][1] > 170:
                dectedby2.append(i)
                print('No.', i, ': bad(detected by 2)')
        else:
            print('No.', i, ':This mosquito\'s mouth is not able to be detected!')
    print('the detected by only one rate is ' + str(1 - len(dectedby2) / mosquitoesnum))
    return dectedby2, ImgAfterClustering, pointsOfAllMos

'''def dbscanAndCovarianceForSoloHead(HeadImg, rank):
    # input:
    #       HeadImg: only the img of the head
    #       rank: it is for the filename. More detail in function -- 'StoreSeperateImg'
    # output:
    #    detectedby2: if the head is detected by 2 location where there are noisy points, it is True.
    #    num_pointInlable: it is a dict, {'-1': [a2, b2, c2,d2], '0': [a1, b1, c1, d1]}
    #
    #
    dectedby2 = False
    Storepath = os.path.join(os.getcwd(), 'afterclustering\\Head')
    HeadImgShape = HeadImg.shape
    db, points = dbscanFromIMG(HeadImg, 0.5, 100)  # Clustering
    X = points
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels))
    print('after 1st clustering, the num of all clusterings: ', n_clusters_)
    num_pointInlable = gt_num_pointInlable(labels)
    for key in num_pointInlable.keys():
        print('cluster: ', key, 'num of points:', len(num_pointInlable[key]))
        StoreSeperateImg(HeadImgShape, points, np.array(num_pointInlable[key]), rank, key, Storepath)
    ##Covariance
    X2Cov = list()
    Y2Cov = list()
    for a in num_pointInlable[str(-1)]:
        X2Cov.append(X[a][0])
        Y2Cov.append(X[a][1])
    X2Cov = np.array(X2Cov)
    Y2Cov = np.array(Y2Cov)
    NoisyCov = np.cov(X2Cov, Y2Cov)
    print('Covariance:\n', NoisyCov)
    if NoisyCov[0][0] < 100 and NoisyCov[1][1] < 100:
        dectedby2 = True
    return dectedby2, num_pointInlable
'''


def JudgeNoisyPointsAfterClustering(pointsOfAllMos, ImgAfterClustering, rectlist, rectForSoloImg, IfStore=False):
    # =============================================================================================
    # To judge the noisy points if in the head bounding box, if so, output those points and store.
    # OUTPUT:NoisyPointsAfterJudgeIfInHeadBBoxForWholeImg = {0: [points]
    #                                                        1: [points]
    #                                                        ...
    #                                                                       }
    # =============================================================================================
    # ImgAfterClustering = dict(ImgAfterClustering)
    NoisyPointsAfterJudgeIfInHeadBBoxForWholeImg = dict()  # store the filtered points in whole image.
    if IfStore:
        folder1 = os.getcwd() + '\\SavaImages\\afterclustering'
        folder11 = os.path.join(folder1, 'MouthAfterFilter\\')
        # 获取此py文件路径，在此路径选创建在new_folder文件夹中的test文件夹
        folder2 = os.path.join(folder11, 'current\\')
        if not os.path.exists(folder1):
            os.makedirs(folder1)
        if not os.path.exists(folder11):
            os.makedirs(folder11)
        if not os.path.exists(folder2):
            os.makedirs(folder2)
        filelist = glob.glob(os.path.join(folder2, "*.jpg"))
        for f in filelist:
            os.remove(f)
    for mos in range(0, len(pointsOfAllMos)):
        mosi = mos * 2 + 1
        mosString = str(mos)
        NoisyPointsAfterJudgeIfInHeadBBox = list()  # to store filtered points for every mos
        if str(-1) in ImgAfterClustering[mosString].keys():
            for point in ImgAfterClustering[mosString]['-1']:
                if pointsOfAllMos[mosString][point, 0] + rectForSoloImg[mos][0] > rectlist[mosi][
                    0]:  # The point X coordinate in whole image
                    if pointsOfAllMos[mosString][point, 0] + rectForSoloImg[mos][0] < rectlist[mosi][
                        2]:  # The point X coordinate in whole image
                        if pointsOfAllMos[mosString][point, 1] + rectForSoloImg[mos][1] > rectlist[mosi][
                            1]:  # The point Y coordinate in whole image
                            if pointsOfAllMos[mosString][point, 1] + rectForSoloImg[mos][1] < rectlist[mosi][
                                3]:  # The point Y coordinate in whole image
                                NoisyPointsAfterJudgeIfInHeadBBox.append(point)
            NoisyPointsAfterJudgeIfInHeadBBoxForWholeImg[mos] = NoisyPointsAfterJudgeIfInHeadBBox
            if NoisyPointsAfterJudgeIfInHeadBBox is []:
                print('No.', + mos + ': the potential mouth is not in Head box')
        else:
            print('No.' + str(mos) + ': there is no potential mouth detected')
        if IfStore:
            Storepath = folder2
            soloimgShape = (
            rectForSoloImg[mos][2] - rectForSoloImg[mos][0], rectForSoloImg[mos][3] - rectForSoloImg[mos][1])
            StoreSeperateImg(soloimgShape, pointsOfAllMos[str(mos)], np.array(NoisyPointsAfterJudgeIfInHeadBBox), mos,
                             -1, Storepath)

    return NoisyPointsAfterJudgeIfInHeadBBoxForWholeImg


def getHeadAndTailRect(erodedpadding, imagename, IfStore=False):
    """
    现在在这张蚊子上测试没有问题，输入为这张蚊子的图片，
    需要的输出在rectlist这个变量上，顺序是    尾巴  头  尾巴  头  尾巴  头  尾巴 头  ……
    四个数分别为xmin,ymin,xmax,ymax,稍微调了一下robustness，
    有些身体中间没有断开的直接返回原始框，
    唯一需要保证的一个条件就是拍摄蚊子时照相机的高度，
    因为我断开脚的时候有一个阈值，不然就要调整代码第二十五行核的参数
    @author: acer
    """
    output = cv2.connectedComponentsWithStats(erodedpadding, 4, cv2.CV_32S)
    num, index, mosquitoesnum, afternoise, rect = getRemovePoints(output[1], output[0], erodedpadding)

    retval1, afternoise = cv2.threshold(afternoise, 10, 255, cv2.THRESH_BINARY)
    rectlist = []
    if IfStore:  ## Clear all files in Current
        folder1 = os.getcwd() + '\\SavaImages\\CropHead'
        # 获取此py文件路径，在此路径选创建在new_folder文件夹中的test文件夹
        folder2 = os.path.join(folder1, 'current')
        folder3 = os.path.join(folder1, 'history')

        if not os.path.exists(folder1):
            os.makedirs(folder1)
        if not os.path.exists(folder2):
            os.makedirs(folder2)
        if not os.path.exists(folder3):
            os.makedirs(folder3)
        filelist = glob.glob(os.path.join(folder2, "*.jpg"))
        for f in filelist:
            os.remove(f)
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

            b = rect[i][0] + rect0[0][0] -3 - 2
            rect0a.append(b)
            b = rect[i][1] + rect0[0][1] -3 - 2
            rect0a.append(b)
            b = rect[i][0] + rect0[0][2] -3 + 2
            rect0a.append(b)
            b = rect[i][1] + rect0[0][3] -3 + 2
            rect0a.append(b)
            rectlist.append(rect0a)

            b = rect[i][0] + rect1[0][0] -3 - 2
            rect1a.append(b)
            b = rect[i][1] + rect1[0][1] -3 - 2
            rect1a.append(b)
            b = rect[i][0] + rect1[0][2] -3 + 2
            rect1a.append(b)
            b = rect[i][1] + rect1[0][3] -3 + 2
            rect1a.append(b)
            rectlist.append(rect1a)
            if IfStore:
                cropimageTail = cropimage[rect0[0][0]-1:rect0[0][2]+1, rect0[0][1]-1:rect0[0][3]+1]
                cropimageHead = cropimage[rect1[0][0]-1:rect1[0][2]+1, rect1[0][1]-1:rect1[0][3]+1]
                cropimage3 = Image.fromarray(cropimageTail)
                cropimage3.save(folder2 + '\\cropimage' + str(i) + '_onlytail.jpg')
                cropimage3.save(folder3 + '\\' + imagename + 'cropimage' + str(i) + '_onlytail.jpg')
                cropimage4 = Image.fromarray(cropimageHead)
                cropimage4.save(folder2 + '\\cropimage' + str(i) + '_onlyhead.jpg')
                cropimage4.save(folder3 + '\\' + imagename + 'cropimage' + str(i) + '_onlyhead.jpg')
    return rectlist


def HolesFill(im_at_fixedOverGrey):
    # 内部孔填充算法，从（0,0）开始
    im_floodfill = im_at_fixedOverGrey.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_at_fixedOverGrey.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    imgOverGreyAfterFill = im_at_fixedOverGrey | im_floodfill_inv
    return imgOverGreyAfterFill


# Hough Ciecles, but the parameter is a exact number.we can modify it in the function
def HoughCircles(img):
    IfPlay = False
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 35,
                               param1=20, param2=8, minRadius=5, maxRadius=13)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print('Potential Head detected, position: ' + 'X:' +  str(circles[0][0][1]) + ', Y:' + str(circles[0][0][2]) + '\n')
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
        if IfPlay:
            cv2.imshow('detected circles', cimg)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

    return circles


def plotTheFeatures(imgColor, CirclesOfAllMos, rectlist, rectForSoloImg, NoisyPointsAfterJudgeIfInHeadBBoxForWholeImg):
    for mos in range(0, len(rectForSoloImg)):
        ## draw Heads Circles
        if CirclesOfAllMos[str(mos)] is not None:
            CircleCentX = CirclesOfAllMos[str(mos)][0][0][1] + rectlist[2 * mos + 1][0]
            CircleCentY = CirclesOfAllMos[str(mos)][0][0][0] + rectlist[2 * mos + 1][1]
            Radius = CirclesOfAllMos[str(mos)][0][0][2]
            rr, cc = draw.circle(CircleCentX, CircleCentY, Radius)
            draw.set_color(imgColor, [rr, cc], [0, 255, 200])
        ## draw Heads and Tails Rects
        HeadRectStart = [rectlist[2 * mos + 1][0], rectlist[2 * mos + 1][1]]
        HeadRectExtent = [rectlist[2 * mos + 1][2], rectlist[2 * mos + 1][3]]
        TailRectStart = [rectlist[2 * mos][0], rectlist[2 * mos][1]]
        TailRectExtent = [rectlist[2 * mos][2], rectlist[2 * mos][3]]
        SoloMosRectStart = [rectForSoloImg[mos][0], rectForSoloImg[mos][1]]
        SoloMosRectExtent = [rectForSoloImg[mos][2], rectForSoloImg[mos][3]]

        # rrHead, ccHead = draw.rectangle(HeadRectStart, HeadRectExtent)
        # draw.set_color(imgColor, [rrHead, ccHead], [0, 255, 0])
        # rrTail, ccTail = draw.rectangle(TailRectStart, TailRectExtent)
        # draw.set_color(imgColor, [rrTail, ccTail], [255, 0, 0])
        # rrSoloMos, ccSoloMos = draw.rectangle(SoloMosRectStart, SoloMosRectExtent)
        # draw.set_color(imgColor, [rrSoloMos, ccSoloMos], [0, 0, 255])
    plt.imshow(imgColor)
    plt.show()
