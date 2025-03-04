import cv2 as cv
import sys

img = cv.imread('img\mong.jpg')

if img is None :
    sys.exit('파일이 존재하지 않습니다.')

def draw(event,x,y,flags,param):
    global ix,iy,roi

    if event==cv.EVENT_LBUTTONDOWN:
        ix,iy=x,y

    elif event==cv.EVENT_LBUTTONUP:
        cv.rectangle(img,(ix,iy),(x,y),(0,0,255),2)
        roi=img[iy:y, ix:x].copy()
        cv.imshow('Drawing',img)
        if roi is not None:
            cv.imshow('ROI',roi)

cv.namedWindow('Drawing')
cv.imshow('Drawing',img)
cv.setMouseCallback('Drawing',draw)

while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break
    elif cv.waitKey(1)==ord('r'):
        img=cv.imread('img\mong.jpg')
        cv.imshow('Drawing',img)
    elif cv.waitKey(1)==ord('s') and roi is not None:
        cv.imwrite('output\ROI.jpg', roi)
        print('저장되었습니다.')