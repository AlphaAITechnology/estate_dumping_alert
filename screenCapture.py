import cv2 as cv


cap = cv.VideoCapture("rtsp://admin:12345678a@180.188.143.227:580 ! decodebin ! videoconvert ! appsink max-buffers=1 drop=trueqqq")
ret, frame = cap.read()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv.imwrite("test_base_3.png", frame)
        cap.release()
        break