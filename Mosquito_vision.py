# -*-coding:utf-8-*-
"""
This is the whole code for the traditional method to do the detection for mosquitoes

@@author: Zeyu Lu, Guanqun Huang
"""
import cv2
import os
import MosquitoClass

def ChooseSource(FromVideo = False, FromImage = False):
    if FromImage:
        datapath = r'C:\Users\Zed_Luz\OneDrive\3-MEE\18-JHU\12-Work\5-MosquitoRecog\7-data\train'
        imglist = os.listdir(datapath)
        imagename = imglist[275]
        imgpath = os.path.join(datapath, imagename)
        imgColor = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    if FromVideo:
        capture = cv2.VideoCapture('ms_test4.mp4')
        i = -1
        frames = dict()
        while True:
            ret, rawfram = capture.read()
            i = i + 1
            frames[i] = rawfram
            print('This is Frame' + str(i) + '\n')
            if len(frames) == 200:
                break

        fram = frames[100]
        imgColor = cv2.cvtColor(fram, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
        imagename = 'Frame' + str(i)
    return imgColor, img, imagename

imgColor, img, imagename = ChooseSource(FromVideo=False, FromImage=True)

# grayprocess, erode using kernel
retval, im_at_fixed = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)# for video, the weo paras are 60/30, for image, they are 110/60
retval2, im_at_fixedOverGrey = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # errosion for remove legs
eroded = cv2.erode(im_at_fixed, kernel)
# padding 0
erodedpadding = cv2.copyMakeBorder(eroded, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)

# main function
output = cv2.connectedComponentsWithStats(erodedpadding, 4, cv2.CV_32S)
ArrayAfterComponents = output[1]
NumOfComponents = output[0]
num, index, mosquitoesnum, afternoise, rectForSoloImg = MosquitoClass.getRemovePoints(ArrayAfterComponents, NumOfComponents,
                                                                        erodedpadding)

# num: n*2 array; index: dict{'1':[x,y]...} mosquitoesnum  ; afternoise: img after removal noisy; rect: array of bbp.
# get soloimage: soloImage[0]->arrayMos1  soloImage[1]->arrayMos2   ...
soloImage = MosquitoClass.SeperateAndStoreInList(afternoise, mosquitoesnum, rectForSoloImg, imagename, IfStore=True)
detectedby2, ImgAfterClustering, pointsOfAllMos = MosquitoClass.dbscanAndCovarianceForSoloMos(soloImage, mosquitoesnum)
rectlist = MosquitoClass.getHeadAndTailRect(erodedpadding, imagename, IfStore=True)
## Huang Guanqun xiajibaluangao
for a in range(0,len(rectForSoloImg)):
    for i in range(0, len(rectForSoloImg[a])):
        rectForSoloImg[a][i] = rectForSoloImg[a][i]-3
NoisyPointsAfterJudgeIfInHeadBBoxForWholeImg = MosquitoClass.JudgeNoisyPointsAfterClustering(
    pointsOfAllMos, ImgAfterClustering,
    rectlist, rectForSoloImg, IfStore=True)
imgOverGreyAfterFill = MosquitoClass.HolesFill(im_at_fixedOverGrey)

CirclesOfAllMos = dict()  # Store the infomation of hough circles, {'0': array([x,y,r])
#                                                                       ...
#                                                                                   }
print('\nStart Hough Circle')
for i in range(0, mosquitoesnum):
    HeadimgForHough = imgOverGreyAfterFill[MosquitoClass.relu(rectlist[2 * i + 1][0]):MosquitoClass.relu(rectlist[2 * i + 1][2]),
                      MosquitoClass.relu(rectlist[2 * i + 1][1]):MosquitoClass.relu(rectlist[2 * i + 1][3])]
    print('\nNo.' + str(i) + ':')
    circles = MosquitoClass.HoughCircles(HeadimgForHough, IfPlay=False)

    CirclesOfAllMos[str(i)] = circles
MosquitoClass.plotTheFeatures(imgColor, CirclesOfAllMos, rectlist, rectForSoloImg, NoisyPointsAfterJudgeIfInHeadBBoxForWholeImg, imagename, Ifsave = True)
