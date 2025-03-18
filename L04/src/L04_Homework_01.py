import cv2 as cv
import sys
import matplotlib.pyplot as plt

img=cv.imread('L04\img\soccer.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

t,bin_img=cv.threshold(gray,127,255,cv.THRESH_BINARY)

h=cv.calcHist([bin_img],[0],None,[256],[0,256]) 
plt.plot(h,color='r',linewidth=1)
plt.show()
