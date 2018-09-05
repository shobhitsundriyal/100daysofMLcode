import urllib
import cv2
import numpy as np
import requests #without this line 7 is not working

url = 'http://192.168.1.4:8080/shot.jpg'
face_cascade = cv2.CascadeClassifier('../100daysofMLcode/Day 16-/haarcascade_frontalface_default.xml')
video = cv2.VideoWriter('../100daysofMLcode/Day 16-/opencv.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280,720), 1)
while True:
    frame = urllib.request.urlopen(url)
    frame_np = np.array(bytearray(frame.read()))
    image = cv2.imdecode(frame_np, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35,35))
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (220,20,60), 2)
    video.write(image)
    cv2.imshow('Test', image)#
    if ord('q') == cv2.waitKey(1):
        break
