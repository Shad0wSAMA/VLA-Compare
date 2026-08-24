import cv2
import time

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS,30)

count = 0
start=time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    count += 1

    if time.time()-start > 10:
        print("FPS:", count/10)
        break