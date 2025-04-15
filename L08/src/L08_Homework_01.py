import numpy as np
import cv2 as cv
import sys
from deep_sort_realtime.deepsort_tracker import DeepSort

def construct_yolo_v4():
    f=open('coco_names.txt', 'r')
    class_names=[line.strip() for line in f.readlines()]
    
    model = cv.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = model.getLayerNames()
    out_layers = [layer_names[i-1] for i in model.getUnconnectedOutLayers()]

    return model, out_layers, class_names

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

model,out_layers,class_names=construct_yolo_v4()
colors=np.random.uniform(0,255,size=(100,3))

sort=DeepSort(max_age=30)
cap=cv.VideoCapture('img/slow_traffic_small.mp4')

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

cap.release()
cv.destroyAllWindows()