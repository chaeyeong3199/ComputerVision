import cv2 as cv
import sys
import numpy as np

img = cv.imread('img\mong.jpg')

if img is None :
    sys.exit('파일이 존재하지 않습니다.')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 컬러를 흑백으로 변환
imgs=np.hstack((img,cv.cvtColor(gray,cv.COLOR_GRAY2BGR)))

cv.imshow('Collected images',imgs)
cv.waitKey()
cv.destroyAllWindows()