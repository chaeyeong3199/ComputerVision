import cv2 as cv
import sys
import matplotlib.pyplot as plt

img=cv.imread('L04\img\soccer.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

t,bin_img=cv.threshold(gray,127,255,cv.THRESH_BINARY)

h=cv.calcHist([bin_img],[0],None,[256],[0,256]) 
g=cv.calcHist([gray],[0],None,[256],[0,256]) 

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(g,color='r',linewidth=1)
plt.title("Grayscale Histogram")
plt.subplot(1,2,2)
plt.plot(h,color='b',linewidth=1)
plt.title("Binary Histogram")
plt.show()
