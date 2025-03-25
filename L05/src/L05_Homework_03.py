import cv2 as cv
import sys
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('L05\img\coffee cup.jpg')

if img is None:
    sys.exit('파일이 존재하지 않습니다.')

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
mode = cv.GC_INIT_WITH_RECT
iterCount = 1
rc = (200, 100, 920, 730)

cv.grabCut(img, mask, rc, bgdModel, fgdModel, iterCount, mode)
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),0,1).astype('uint8')
dst = img * mask2[:, :, np.newaxis]

fig, axes = plt.subplots(1, 3, figsize=(15,5))
axes[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(mask2, cmap='gray')
axes[1].set_title("Mask Image")
axes[1].axis("off")

axes[2].imshow(cv.cvtColor(dst, cv.COLOR_BGR2RGB))
axes[2].set_title("Background Remove Image")
axes[2].axis("off")
plt.show()