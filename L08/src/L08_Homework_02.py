import cv2 as cv
import mediapipe as mp

mp_mesh = mp.solutions.face_mesh

mesh = mp_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, 
                        min_detection_confidence=0.5,min_tracking_confidence=0.5)

cap = cv.VideoCapture('img/face.mp4')

fps = cap.get(cv.CAP_PROP_FPS)
delay = int(1000 / fps) 

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 획득에 실패하여 루프를 나갑니다.")
        break

    res = mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if res.multi_face_landmarks:
        for landmarks in res.multi_face_landmarks:
            for idx, lm in enumerate(landmarks.landmark):
                ih, iw, _ = frame.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                cv.circle(frame, (x, y), 1, (255, 0, 0), -1)

    cv.imshow('Face Landmarks', frame)

    if cv.waitKey(delay) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
