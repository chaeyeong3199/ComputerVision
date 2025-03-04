import cv2 as cv
import sys
import numpy as np

cap=cv.VideoCapture(0,cv.CAP_DSHOW)

if not cap.isOpened():
    sys.exit('카메라 연결 실패')

while True:
    ret,frame=cap.read()

    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    edges=cv.Canny(gray,100,100)

    imgs=np.hstack((frame, cv.cvtColor(edges,cv.COLOR_GRAY2BGR)))

    cv.imshow('Video display', imgs)

    key=cv.waitKey(1)
    if key==ord('q'):
        break
 
cap.release()
cv.destroyAllWindows()

