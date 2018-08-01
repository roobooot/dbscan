import cv2
import numpy as np
for i in range(0,19):
    img = cv2.imread('edcurrent\\cropimage'+str(i)+'_onlyhead.jpg', 0)
    print('edcurrent\\cropimage'+str(i)+'_onlyhead.jpg')
    print('\n',img.shape)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 35,
                               param1=20, param2=8, minRadius=5, maxRadius=13)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(circles.shape,'\n',len(img))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imshow('detected circles', cimg)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
