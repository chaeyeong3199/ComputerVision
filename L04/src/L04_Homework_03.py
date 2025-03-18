import cv2 as cv
import sys
import numpy as np

img=cv.imread('tree.png')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

rows, cols = img.shape[:2]
center = (cols / 2, rows / 2)

matrix = cv.getRotationMatrix2D(center, 45, 1.5)
change = cv.warpAffine(img, matrix, (int(cols*1.5),int(rows*1.5)), flags=cv.INTER_LINEAR)

cv.imshow('Original image', img)
cv.imshow('Change image', change)
cv.waitKey()
cv.destroyAllWindows()