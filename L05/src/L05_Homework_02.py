import cv2 as cv
import sys
import numpy as np 
import matplotlib.pyplot as plt

img=cv.imread('L05\img\dabotop.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

edge=cv.Canny(img,100,200)
lines=cv.HoughLinesP(edge, 1, np.pi/180, 80, minLineLength=15, maxLineGap=5)

line_img=img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_img, (x1, y1), (x2, y2), (0,0,255), 2)

fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(cv.cvtColor(line_img, cv.COLOR_BGR2RGB))
axes[1].set_title("Line Image")
axes[1].axis("off")
plt.show()