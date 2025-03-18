import cv2 as cv
import sys
import numpy as np 
import matplotlib.pyplot as plt

img=cv.imread('L04\img\JohnHancocksSignature.png',cv.IMREAD_UNCHANGED)

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

t,bin_img=cv.threshold(img[:,:,3],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
b=bin_img[bin_img.shape[0]//2:bin_img.shape[0],0:bin_img.shape[0]//2+1]

k = cv.getStructuringElement(cv.MORPH_RECT,(5,5))

dilation = cv.morphologyEx(b, cv.MORPH_DILATE, k)
erosion = cv.morphologyEx(b, cv.MORPH_ERODE, k)
open = cv.morphologyEx(b, cv.MORPH_OPEN, k)
close = cv.morphologyEx(b, cv.MORPH_CLOSE, k)

imgs=np.hstack((b, dilation, erosion, open, close))

cv.imshow('images',imgs)
cv.waitKey()
cv.destroyAllWindows()