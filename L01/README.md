# 📌 L01: OpenCV Homework

## 📝 과제 내용

### 1. 이미지 불러오기 및 그레이스케일 변환

   1. OpenCV를 사용하여 이미지를 불러오고 화면에 출력
   ```python
   img=cv.imread('img\mong.jpg')
   ```
   2. 그레이스케일 변환
   ```python
   gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
   ```
   3. 흑백 이미지를 다시 컬러로 변환하여 연결
   ```python
   imgs=np.hstack((img,cv.cvtColor(gray,cv.COLOR_GRAY2BGR))) 
   ```
   <img src="output/img_gray.jpg" width="400" height="150">
     
### 2. 웹캠 영상에서 에지 검출
   1. 웹캠을 사용하여 실시간 비디오스트림을 가져옴
   ```python
   cap=cv.VideoCapture(0,cv.CAP_DSHOW)
   ```
   2. 그레이스케일 변환 및 에지 검출
   ```python
   gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
   edges=cv.Canny(gray,100,100)
   ```
   3. 원본영상과 함께 출력
   ```python
   imgs=np.hstack((frame, cv.cvtColor(edges,cv.COLOR_GRAY2BGR)))
   cv.imshow('Video display', imgs)
   ```
   <img src="output/gray.jpg" width="400" height="150">
   
### 3. 마우스로 ROI(관심영역) 선택 및 저장
   1. 이미지 로드
   ```python
   img = cv.imread('img\mong.jpg')
   ```
   2. 마우스 이벤트 함수 정의
      1) 마우스 왼쪽 버튼이 클릭되면 (ix, iy)로 클릭 좌표를 저장
      ```python
      if event==cv.EVENT_LBUTTONDOWN:
          ix,iy=x,y
      ```
      2) 마우스 왼쪽 버튼을 떼면, 이전 좌표와 현재 좌표를 사용해 ROI 선택
      ```python
       elif event==cv.EVENT_LBUTTONUP:
           cv.rectangle(img,(ix,iy),(x,y),(0,0,255),2)
           roi=img[iy:y, ix:x].copy()
           cv.imshow('Drawing',img)
           if roi is not None:
               cv.imshow('ROI',roi)
      ```
   3. 키 입력 처리
      1) r키를 누르면 영역선택을 리셋하고 처음부터 다시선택
      ```python
       elif cv.waitKey(1)==ord('r'):
           img=cv.imread('img\mong.jpg')
           cv.imshow('Drawing',img)
      ```
      2) s키를 누르면 선택한 영역을 이미지 파일로 저장
      ```python
       elif cv.waitKey(1)==ord('s') and roi is not None:
           cv.imwrite('output\ROI.jpg', roi)
           print('저장되었습니다.')
      ```
   <img src="output/ROI_result.jpg" width="300" height="200">
