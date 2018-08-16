def find_head_v1(img, cimg, bb_list, min_dist=35, acc_thresh=8, remove_body_thresh=80):
    #using hough circle detection
    
    #remove_body    
    img, diff = remove_body_head(img, remove_body_thresh)
    diff = 255-diff
    img[diff==255] = 255
    
    #img_s = img.copy()
    #img_s = cv2.resize(img_s,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    #cv2.imshow('body_thresh_img', img_s)

    ret, thresh = cv2.threshold(img,45,255,cv2.THRESH_BINARY)
    img[thresh>0] = 255
    img = cv2.medianBlur(img,5)
    
    img_s = img.copy()
    img_s = cv2.resize(img_s,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('body_thresh_img', img_s)
    
    head_list = list()
    for (x,y,w,h) in bb_list:
        roi = img[y:y+h, x:x+w]
#         cv2.rectangle(cimg,(x,y),(x+w,y+h),(255,0,255),2)
        circles = cv2.HoughCircles(roi,cv2.HOUGH_GRADIENT,1,minDist = min_dist,
                                    param1=20,param2=acc_thresh,minRadius=5,maxRadius=13)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            aver_min = 9999
            for i in circles[0,:]:
                # draw the outer circle
                dist_list = [i[0], abs(w-i[0]), i[1], abs(h-i[1])]
                min_dist = min(dist_list)
#                 if min_dist<20:
                aver = get_img_value(img, x + i[0], y + i[1])
                if aver<aver_min:
                    aver_min = aver
                    c_x = x + i[0]
                    c_y = y + i[1]

            if aver_min<120:
                cv2.circle(cimg,(c_x,c_y),10,(0,255,0),2)
                head_list.append((c_y, c_x))
                    # draw the center of the circle
        #             cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
    return head_list