# 📌 L07: Recognition

## 1. 간단한 이미지 분류기 구현

### 1. 데이터 로드 및 전처리
   
   ```python
   (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
   x_train = x_train.astype("float32") / 255.0
   x_test = x_test.astype("float32") / 255.0
   y_train = utils.to_categorical(y_train)
   y_test = utils.to_categorical(y_test)
   ```
   - TensorFlow에서 제공하는 MNIST 데이터셋을 사용
   - 훈련 세트와 테스트 세트로 분할
   - 픽셀 값을 0~1 범위로 정규화, 원-핫 인코딩

### 2. 신경망 모델 구축

   ```python
   model = models.Sequential()
   model.add(layers.Input(shape=(28, 28)))
   model.add(layers.Flatten())
   model.add(layers.Dense(units=64, activation='relu'))
   model.add(layers.Dense(units=10, activation='softmax'))
   ```
   - 입력층: 28x28 픽셀의 흑백 이미지
   - 은닉층: Dense(64 units, ReLU)
   - 출력층: Dense(10 units, Softmax) – 10개의 숫자 분류

### 3. 모델 컴파일 및 학습
   ```python
   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   history = model.fit(x_train, y_train, epochs=7, batch_size=64, verbose=1)
   ```
   - Optimizer: Adam
   - Loss Function: categorical_crossentropy
   - Metric: Accuracy
   - Epochs: 7
   - Batch Size: 64

### 4. 정확도 평가
   ```python
   loss, acc = model.evaluate(x_test, y_test, verbose=0)
   print(f"정확도: {acc:.4f}")
   ```
![output1-1](https://github.com/user-attachments/assets/4d74ac9c-2a2a-4bb4-96eb-1b0f8a763fd9)

  #### 결과이미지
![output1-2](https://github.com/user-attachments/assets/02c81197-e90c-4da0-a01c-81a3fa8cc25f)
<br><br><br>
     
## 2. CIFAR-10 데이터셋을 활용한 CNN 모델 구축

### 1. 데이터 로드 및 전처리
   ```python
   (x_train, y_train), (x_test, y_test) = cifar10.load_data()
   x_train = x_train.astype("float32") / 255.0
   x_test = x_test.astype("float32") / 255.0
   y_train = tf.keras.utils.to_categorical(y_train, 10)
   y_test = tf.keras.utils.to_categorical(y_test, 10)
   ```
   - 데이터셋: CIFAR-10 – 총 60,000개의 32x32 RGB 이미지
   - 픽셀 값을 0~1 범위로 정규화, 원-핫 인코딩

### 2. CNN 모델 구축

   ```python
   model = models.Sequential([
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
      layers.MaxPooling2D((2, 2)),
      layers.Dropout((0.2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.Dropout((0.2)),

      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10, activation='softmax')  
   ])
   ```
   - Conv2D + ReLU: 특징 추출
   - MaxPooling2D: 공간 정보 축소
   - Dropout: 과적합 방지
   - Flatten + Dense: 분류기(Fully Connected Layers) 구성

### 3. 모델 컴파일 및 학습

   ```python
   model.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

   history = model.fit(x_train, y_train, epochs=20,
                     validation_data=(x_test, y_test),
                     batch_size=64)
   ```
  - Optimizer: Adam
  - Loss Function: categorical_crossentropy
  - Metric: Accuracy
  - Epochs: 20
  - Batch Size: 64
  - Validation Set: 테스트셋 사용

### 4. 정확도 평가

   ```python
   test_loss, test_acc = model.evaluate(x_test, y_test)
   print(f"\nTest Accuracy: {test_acc:.4f}")
   ```
![output2-1](https://github.com/user-attachments/assets/54f67e6f-05d0-40bb-9456-02d4ea631890)

  ### 예측결과 시각화 
![output2-2](https://github.com/user-attachments/assets/6fd3f93c-e4d2-4961-a82c-f66e0e5accd0)
<br><br><br>
   
## 3. 전이 학습을 활용한 이미지 분류기 개선

### 1. 데이터 전처리 및 증강
   
   ```python
   train_dir = 'L07/dataset/train'
   val_dir = 'L07/dataset/val'

   datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True
   )

   train_gen = datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=16, class_mode='binary')
   val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_dir, target_size=(224, 224), batch_size=16, class_mode='binary')

   ```
   - 사용자 정의 데이터셋에 대한 이진 분류를 수행
   - 학습/검증 데이터를 ImageDataGenerator를 통해 불러오고 증강
   - 학습 데이터는 회전, 이동, 반전 등을 적용하여 성능을 높임 

### 2-1. VGG 모델 구축

   ```python
   base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
   base.trainable = False
   ```
   - weights='imagenet': ImageNet 데이터셋으로 학습된 가중치 사용
   - include_top=False: VGG16의 최상위 레이어를 제거
   - input_shape=(32, 32, 3): CIFAR-10 이미지에 맞춤
   - trainable=False: 가중치를 고정하여 학습 시간과 과적합을 줄임

### 2-2. VGG 모델 구축

   ```python
   model = Sequential([
      base,
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(1, activation='sigmoid')
   ])
   ```
   - Flatten: VGG16에서 추출한 특징 맵을 1차원으로 변환
   - Dense(128): 은닉층으로 고차원 특징 조합
   - Dropout(0.5): 학습 시 50% 뉴런 무작위 비활성화
   - Dense(1, activation='sigmoid'): sigmoid로 이진 분류 수행

### 3. 모델 컴파일 및 학습
   ```python
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   history = model.fit(train_gen, epochs=10, validation_data=val_gen)
   ```
   - 손실 함수: binary_crossentropy
   - optimizer: adam
   - 평가지표: accuracy
   - Epochs: 10

### 4. 정확도 평가
   ```python
   loss, acc = model.evaluate(x_test, y_test, verbose=0)
   print(f"정확도: {acc:.4f}")
   ```
- VGG 모델 정확도 <br>

![image](https://github.com/user-attachments/assets/6468d63c-f32c-450d-9bc7-051c1af32325)
- CNN 모델 정확도 <br>

![image](https://github.com/user-attachments/assets/aed0db98-ed93-4f98-a8d7-a082f7d7fdbd)

### 결과 이미지 (VGG vs CNN)
![image](https://github.com/user-attachments/assets/b627f294-120d-46bb-a48c-5930555f4a82) ![image](https://github.com/user-attachments/assets/de9e2d03-7396-493a-9c93-c87d3c9ac962)


