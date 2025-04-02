# 📌 L05: Edge and Region Homework

## 📝 과제 내용

### 1. 소벨 에지 검출 및 결과 시각화
   1. OpenCV를 사용하여 이미지를 불러옴
   ```python
  img=cv.imread('L05\img\edgeDetectionImage.jpg')
   ```
   2. 그레이스케일 변환
   ```python
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   ```
   3. 소벨(Sobel) 필터를 사용하여 X축과 Y축 방향의 에지를 검출

   ```python
  grad_x=cv.Sobel(gray,cv.CV_64F,1,0,ksize=3)
  grad_y=cv.Sobel(gray,cv.CV_64F,0,1,ksize=3)
   ```
   - **cv.Sobel(src, ddepth, dx, dy, ksize)** : Sobel 필터 적용
   - cv.CV_64F: 결과를 부동소수점 형식(64비트)으로 반환

   4. 검출된 에지 강도(edge strength) 이미지를 시각화
   ```python
  magnitude=cv.magnitude(grad_x,grad_y)
  edge_strength=cv.convertScaleAbs(magnitude)
   ```
   - **cv.magnitude(grad_x, grad_y)**: 에지 강도를 계산
   - **cv.convertScaleAbs(src)**: 절대값을 취한 후 8비트 정수형으로 변환

  <details>
     <summary>전체코드</summary>
     
   ```python
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
   ```
  </details>
  
  #### 결과이미지
   <img src="output/Edge_Strength.jpg" width="700" height="300">
     
### 2. 캐니 에지 및 허프 변환을 이용한 직선 검출
   1. 캐니(Canny) 에지 검출을 사용하여 에지 맵을 생성
   ```python
  edge=cv.Canny(img,100,200)
   ```
   
   2. 허프 변환(Hough Transform)을 사용하여 이미지에서 직선을 검출
   ```python
  lines=cv.HoughLinesP(edge, 1, np.pi/180, 80, minLineLength=15, maxLineGap=5)
   ```
   - rho(거리해상도)=1, theta(각도해상도)=np.pi/180=1,
   - threshold(직선으로 간주할 최소 교차점 개수)=80, minLineLength=15, maxLineGap=5

   3. 검출된 직선을 원본 이미지에 빨간색(0,0,255)으로 표시
   ```python
  line_img=img.copy()
  if lines is not None:
      for line in lines:
          x1, y1, x2, y2 = line[0]
          cv.line(line_img, (x1, y1), (x2, y2), (0,0,255), 2)
   ```

  <details>
     <summary>전체코드</summary>
     
   ```python
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
   ```
  </details>
  
  #### 결과이미지 
   <img src="output/Line.jpg" width="750" height="300">
   
### 3. GrabCut을 이용한 대화식 영역 분할 및 객체 추출
   1. GrabCut parameter 설정
   ```python
  mask = np.zeros(img.shape[:2], np.uint8)
  bgdModel = np.zeros((1, 65), np.float64)
  fgdModel = np.zeros((1, 65), np.float64)
  mode = cv.GC_INIT_WITH_RECT
  iterCount = 1
  rc = (200, 100, 920, 730)
   ```
   - mask = np.zeros(img.shape[:2], np.uint8) : 이미지 크기와 같은 0으로 초기화된 마스크 생성
   - bgdModel과 fgdModel을 np.float64 타입으로 초기화
   - cv.GC_INIT_WITH_RECT : 사각형모드로 초기화
   - 1번 반복하여 Grabcut수행
   - 사각형 영역 (x=200, y=100, width=920, height=730) 지정
  
   2. Grapcut 알고리즘 수행
   ```python
  cv.grabCut(img, mask, rc, bgdModel, fgdModel, iterCount, mode)
   ```

   3. 마스크를 사용해 원본 이미지에서 배경을 제거
   ```python
  mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),0,1).astype('uint8')
  dst = img * mask2[:, :, np.newaxis]
   ```
   - mask 값이 cv.GC_BGD(0), cv.GC_PR_BGD(2)인 픽셀은 배경으로 간주하여 제거(0으로 설정)
   - mask2[:, :, np.newaxis] : 배경을 제거하고 전경(객체)만 남긴 새로운 이미지를 생성
  <details>
     <summary>전체코드</summary>
     
   ```python
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
   ```
  </details>

  #### 결과이미지
   <img src="output/mask.jpg" width="800" height="300">
