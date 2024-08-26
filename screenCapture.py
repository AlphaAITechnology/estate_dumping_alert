import cv2 as cv
import datetime


cap = cv.VideoCapture("rtsp://admin:hik12345@180.188.143.227:581 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=trueqqq")
ret, frame = cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv.imwrite(f"screen_capture_{datetime.datetime.now().isoformat()}.png", frame)
        cap.release()
        break