import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1=cv.imread('L06\img\img1.jpg')
gray1=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
img2=cv.imread('L06\img\img2.jpg')
gray2=cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

sift=cv.SIFT_create()
kp1,des1=sift.detectAndCompute(gray1,None)
kp2,des2=sift.detectAndCompute(gray2,None)

bf_matcher=cv.BFMatcher(cv.NORM_L2, crossCheck=False)
bf_match=bf_matcher.knnMatch(des1, des2, 2)

T=0.7
good_match=[]
for nearest1,nearest2 in bf_match:
    if (nearest1.distance/nearest2.distance)<T:
        good_match.append(nearest1)

points1=np.float32([kp1[gm.queryIdx].pt for gm in good_match])
points2=np.float32([kp2[gm.trainIdx].pt for gm in good_match])

H, mask = cv.findHomography(points1, points2, cv.RANSAC)

h1, w1 = img1.shape[0],img1.shape[1]
warp= cv.warpPerspective(img2, H, (w1, h1))
img_match=cv.drawMatches(img1,kp1,img2,kp2,good_match,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

fig, axes = plt.subplots(1, 3, figsize=(20,5))
axes[0].imshow(img1)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(warp)
axes[1].set_title("Warped Image")
axes[1].axis("off")

axes[2].imshow(img_match)
axes[2].set_title("Matching Result")
axes[2].axis("off")

plt.tight_layout()
plt.show()