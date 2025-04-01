import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread('L06\img\mot_color70.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift=cv.SIFT_create(nfeatures=1000)
kp,des=sift.detectAndCompute(gray,None)

gray=cv.drawKeypoints(gray,kp,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig, axes = plt.subplots(1, 2, figsize=(15,5))
axes[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(gray)
axes[1].set_title("SIFT Image")
axes[1].axis("off")
plt.tight_layout()
plt.show()