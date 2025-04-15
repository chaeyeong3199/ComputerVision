# 📌 L08: Dynamic Vision

## 1. SORT 알고리즘을 활용한 다중 객체 추적기 구현

### 1. 사전 학습 모델을 읽어 YOLO 구성
   
   ```python
   def construct_yolo_v4():
      f=open('coco_names.txt', 'r')
      class_names=[line.strip() for line in f.readlines()]
      
      model = cv.dnn.readNet("yolov4.weights", "yolov4.cfg")
      layer_names = model.getLayerNames()
      out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]

      return model, out_layers, class_names
   ```
   
   - OpenCV의 DNN 모듈을 사용하여 YOLOv4 모델을 로드
   - coco_names.txt를 통해 클래스 이름 목록(class_names) 로딩
   - getUnconnectedOutLayers()를 통해 YOLO의 출력 계층(output layers)을 가져와 후처리에 활용

### 2. YOLOv4 모델을 사용해 객체를 검출

   ```python
   def yolo_detect(img,yolo_model,out_layers):
      height,width=img.shape[0],img.shape[1]
      test_img=cv.dnn.blobFromImage(img,1.0/256,(448,448),(0,0,0),swapRB=True)

      yolo_model.setInput(test_img)
      output4=yolo_model.forward(out_layers)

      box,conf,id=[],[],[]
      for output in output4:
         for vec85 in output:
               scores=vec85[5:]
               class_id=np.argmax(scores)
               confidence=scores[class_id]
               if confidence>0.5:
                  centerx,centery=int(vec85[0]*width),int(vec85[1]*height)
                  w,h=int(vec85[2]*width),int(vec85[3]*height)
                  x,y=int(centerx-w/2),int(centery-h/2)
                  box.append([x,y, w, h])
                  conf.append(float(confidence))
                  id.append(class_id)
      
      ind=cv.dnn.NMSBoxes(box,conf,0.5,0.4)
      results = []
      for i in ind:
         x, y, w, h = box[i]
         results.append((x, y, w, h, conf[i], id[i]))

      return results
   ```
   - 입력 이미지를 전처리하여 YOLO 모델에 맞는 형태(blob)로 변환
   - 모델 추론을 통해 검출된 결과 중 confidence(신뢰도)가 0.5 이상인 경우만 필터링
   - NMS(Non-Maximum Suppression)를 사용하여 중복 박스를 제거
   - 최종적으로 [x, y, w, h, conf, id] 형식으로 객체 정보를 반환

### 3. 모델 초기화 및 비디오 캡처 객체 생성
   ```python
   model,out_layers,class_names=construct_yolo_v4()
   colors=np.random.uniform(0,255,size=(100,3))

   sort=DeepSort(max_age=30)
   cap=cv.VideoCapture('img/slow_traffic_small.mp4')
   ```
   - YOLO 모델과 클래스 이름, 출력 계층을 초기화
   - 추적 객체들의 ID마다 고유한 색상을 지정하기 위해 색상 배열 생성
   - 객체의 appearance 정보를 활용하는 Deep SORT 알고리즘을 사용해 추적 성능 향상
   - 입력 비디오를 불러오기 위한 cv.VideoCapture 객체 생성


### 4. 객체 추적 및 결과 시각화
   ```python
   while True:
      ret,frame=cap.read()
      if not ret: sys.exit('프레임 획득에 실패하여 루프를 나갑니다.')

      dets=yolo_detect(frame,model,out_layers)
      deep_sort=[]
      for det in dets:
         x, y, w, h, conf, id = det
         deep_sort.append(([x, y, w, h], conf, class_names[id]))

      tracks=sort.update_tracks(deep_sort, frame=frame)

      for track in tracks:
         if not track.is_confirmed():
               continue
         track_id = track.track_id
         ltrb = track.to_ltrb()
         x1, y1, x2, y2 = map(int, ltrb)
         color = colors[int(track_id) % 100]
         cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
         cv.putText(frame, str(track_id), (x1, y1 - 10), cv.FONT_HERSHEY_PLAIN, 2, color, 2)

      cv.imshow('SORT Tracking', frame)
      
      key=cv.waitKey(1)
      if key==ord('q'): break
   ```
   - 각 프레임마다 YOLO를 통해 객체 검출
   - 검출된 객체들을 Deep SORT에 전달해 ID 기반의 다중 객체 추적 수행
   - 추적된 각 객체에 고유 ID를 부여
   - 해당 ID와 경계 상자를 비디오프레임에 표시하여 실시간으로 출력

  #### 결과이미지
![image](https://github.com/user-attachments/assets/48842f37-4c7c-4f05-8769-f43b44d6d229)

<br><br><br>
     
## 2. Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화

### 1.Mediapipe의 FaceMesh 모듈을 사용해 랜드마크 검출기 초기화
   ```python
   mp_mesh = mp.solutions.face_mesh

   mesh = mp_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                           min_detection_confidence=0.5,min_tracking_confidence=0.5)
   ```
   - solutions.face_mesh를 사용하여 얼굴 랜드마크 검출기를 생성
   - 최대 2명까지 얼굴 탐지 설정 (max_num_faces=2)
   - 눈, 입 등의 정밀 랜드마크 보정을 위해 refine_landmarks=True
   - 검출 및 추적 신뢰도 설정 (min_detection_confidence, min_tracking_confidence)
   
### 2. 영상 불러오기 및 재생 속도 설정

   ```python
   cap = cv.VideoCapture('img/face.mp4')

   fps = cap.get(cv.CAP_PROP_FPS)
   delay = int(1000 / fps) 
   ```
   - OpenCV의 VideoCapture를 사용하여 비디오 파일을 열기
   - cap.get(cv.CAP_PROP_FPS)를 통해 영상의 프레임 속도(FPS)를 확인
   - FPS를 기반으로 프레임 간 대기 시간(delay)을 계산하여 자연스러운 재생 구현

### 3. 프레임 처리 및 랜드마크 검출

   ```python
   while True:
      ret, frame = cap.read()
      if not ret:
         print("프레임 획득에 실패하여 루프를 나갑니다.")
         break

      res = mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
   ```
  - 실시간으로 프레임을 읽어오고 실패 시 루프 종료
  - Mediapipe는 RGB 이미지를 입력으로 받으므로 BGR → RGB로 변환
  - 변환된 이미지를 FaceMesh 모델에 전달하여 랜드마크 추출 수행

### 4. 얼굴 랜드마크 시각화

   ```python
      if res.multi_face_landmarks:
         for landmarks in res.multi_face_landmarks:
               for idx, lm in enumerate(landmarks.landmark):
                  ih, iw, _ = frame.shape
                  x, y = int(lm.x * iw), int(lm.y * ih)
                  cv.circle(frame, (x, y), 1, (255, 0, 0), -1)

      cv.imshow('Face Landmarks', frame)

      if cv.waitKey(1) & 0xFF == 27:
         break
   ```
   - 검출된 각 얼굴에 대해 468개의 얼굴 랜드마크를 순회하며 위치 계산
   - 각 랜드마크 좌표에 OpenCV의 circle 함수를 사용해 작은 원을 그려 시각화
   - 랜드마크 좌표는 정규화되어 있으므로, 이미지 크기에 맞게 픽셀 좌표로 변환
   - ESC 키를 누르면 프로그램이 종료되도록 설정

  ### 결과 이미지 
![image](https://github.com/user-attachments/assets/fc3ae3cb-fcbf-476d-8bed-19b09b4a841f)

<br>
