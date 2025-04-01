# 📌 L06: Local Feature

## 📝 과제 내용

### 1. SIFT를 이용한 특징점 검출 및 시각화
   1. 이미지 로드 및 그레이 스케일 변환
   ```python
  img=cv.imread('L06\img\mot_color70.jpg')
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   ```
   2. SIFT 객체 생성
   - **SIFT(Scale-Invariant Feature Transform)** : 이미지의 크기(scale) 및 회전(rotation)에 영향을 받지 않는 특징점을 추출하는 알고리즘
   - nfeature값(최대 특징점 수)을 1000개로 지정
   ```python
  sift=cv.SIFT_create(nfeatures=1000)
   ```
   3. detectAndCompute()를 통해 특징점 검출
   - **cv2.Feature2D.detectAndCompute(image, mask=None)**: 특징점 검출과 특징 디스크립터 계산을 한 번에 수행하는 함수
   - image: 입력 이미지, mask: 특징점 검출에 사용할 마스크
   ```python
  kp,des=sift.detectAndCompute(gray,None)
   ```
   4. drawKeypoints()를 사용하여 특징점을 이미지에 시각화
   - **cv2.drawKeypoints(image, keypoints, outImage, color=None, flags=None)** 
   - image: 입력 영상, keypoints: 검출된 특징점 정보, outImage: 출력 영상
   - flags: 특징점 표현 방법 -> DEFAULT(위치만 표현),**DRAW_RICH_KEYPOINTS(크기와 방향을 반영)**
   ```python
  gray=cv.drawKeypoints(gray,kp,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
   ```


  #### 결과이미지
   ![alt text](image.png)
     
### 2. SIFT를 이용한 두 영상 간 특징점 매칭
   1. 두 이미지의 SIFT 특징점 추출
   ```python
  sift=cv.SIFT_create()
  kp1,des1=sift.detectAndCompute(gray1,None)
  kp2,des2=sift.detectAndCompute(gray2,None)
   ```
   2. FlannBasedMatcher()을 통해 특징점 매칭
   - **cv2.FlannBasedMatcher(indexParams, searchParams)**: FLANN (Fast Library for Approximate Nearest Neighbors) 기반 Matcher 생성
   - index_params: 특징점 검색을 위한 구조 설정 (KD-Tree 사용)
   - search_params: 검색 과정에서 고려할 이웃 개수 설정
   ```python
   index_params = dict(algorithm=1, trees=5)
   search_params = dict(checks=50)
   flann_matcher=cv.FlannBasedMatcher(index_params, search_params)
   ```
   3. 최근접 이웃 거리 비율을 적용한 KNN 매칭
   - **knnMatch(descripter1,descripter2,k)** : 각 descripter를 비교해 k개의 최근접 이웃을 찾음
   - T=0.7: 거리가 가까운 두 점의 비율이 0.7 이하인 경우 좋은 매칭
   ```python
   knn_match=flann_matcher.knnMatch(des1,des2,2)
   T=0.7
   good_match=[]
   for nearest1,nearest2 in knn_match:
      if (nearest1.distance/nearest2.distance)<T:
         good_match.append(nearest1)
   ```
   4. drawMatches()를 통해 매칭 결과를 시각화
   - np.empty(): 두 이미지를 나란히 배치하기 위해 빈 배열을 생성
   - **cv.drawMatches()**: 매칭된 특징점을 시각적으로 연결하여 출력
   ```python
   img_match=np.empty((max(img1.shape[0],img2.shape[0]),img1.shape[1]+img2.shape[1],3),dtype=np.uint8)
   cv.drawMatches(img1,kp1,img2,kp2,good_match,img_match,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
   ```
  #### 결과이미지 
   ![alt text](image-1.png)
   
### 3. 호모그래피를 이용한 이미지 정합 (Image Alignment)
   1. 두 이미지의 SIFT 특징점 추출
   ```python
   sift=cv.SIFT_create()
   kp1,des1=sift.detectAndCompute(gray1,None)
   kp2,des2=sift.detectAndCompute(gray2,None)
   ```
   2. BFMatcher()를 사용하여 특징점을 매칭
   - **cv.BFMatcher(normType, crossCheck)**: Brute-Force 방식의 Matcher 생성
   - cv.NORM_L2: SIFT와 같은 float형 descripter에 적합한 거리 측정 방식
   - crossCheck=False: KNN 매칭을 사용하므로 필요X
   ```python
   bf_matcher=cv.BFMatcher(cv.NORM_L2, crossCheck=False)
   bf_match=bf_matcher.knnMatch(des1, des2, 2)
   ```
   3. findHomography()를 통해 호모그래피 행렬을 계산
   - np.float32(): 특징점 좌표를 실수형 배열로 변환
   - **cv.findHomography(srcPoints, dstPoints, method)**: RANSAC을 사용하여 변환 행렬 H를 계산 -> outlier 제거
   ```python
   points1=np.float32([kp1[gm.queryIdx].pt for gm in good_match])
   points2=np.float32([kp2[gm.trainIdx].pt for gm in good_match])

   H, mask = cv.findHomography(points1, points2, cv.RANSAC)
   ```
   4. 이미지를 변환하여 다른 이미지와 정렬
   - **cv.warpPerspective(img, H, size)**: 이미지의 투시 변환(perspective transformation)을 수행
   - 출력 크기를 원본 이미지 크기와 동일하게 설정
   ```python
   h1, w1 = img1.shape[0],img1.shape[1]
   warp= cv.warpPerspective(img2, H, (w1, h1))
   ```
   #### 결과이미지 
   ![alt text](image.png)
