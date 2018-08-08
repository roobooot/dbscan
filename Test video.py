import cv2
capture = cv2.VideoCapture('ms_test4.mp4')
i = 0
while True:
    ret, fram = capture.read()
    gray = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)
    i = i+1
    cv2.imshow('fram', gray)
    if cv2.waitKey(1) & i ==200:
        break

capture.release()
cv2.destroyAllWindows()

