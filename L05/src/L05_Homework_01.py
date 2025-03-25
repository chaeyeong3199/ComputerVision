import cv2 as cv
import sys
import matplotlib.pyplot as plt

img=cv.imread('L05\img\edgeDetectionImage.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')
    
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grad_x=cv.Sobel(gray,cv.CV_64F,1,0,ksize=3)
grad_y=cv.Sobel(gray,cv.CV_64F,0,1,ksize=3)

magnitude=cv.magnitude(grad_x,grad_y)
edge_strength=cv.convertScaleAbs(magnitude)

fig, axes = plt.subplots(1, 2, figsize=(10,5))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(edge_strength, cmap='gray')
axes[1].set_title("Edge Strength Image")
axes[1].axis("off")
plt.show()