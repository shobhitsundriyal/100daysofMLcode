import cv2
import numpy as np
import os
import urllib
import requests

image_x, image_y = 50, 50
url = 'http://192.168.1.3:8080/shot.jpg'

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def strore_images(gesture_id):
    total_pics = 1200
    x,y,w,h = 350,50,350,350

    create_folder('../100daysofMLcode/DataSet/Gestures/'+str(gesture_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0
    while True:
        frame_url = urllib.request.urlopen(url)
        frame_np = np.array(bytearray(frame_url.read()))
        frame = cv2.imdecode(frame_np, -1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, np.array([2,50,60]), np.array([25,155,255]))#Skin Color
        res = cv2.bitwise_and(frame, frame, mask=mask2)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray,(5,5),0)

        kernel_square = np.ones((5,5), np.uint8)
        dilation = cv2.dilate(median, kernel_square, iterations=2)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_square)

        ret,thresh = cv2.threshold(opening, 3, 25, cv2.THRESH_BINARY)
        thresh = thresh[y:y+h, x:x+w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

        if len(contours)>0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 50:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh[y1:y1 + h1, x1:x1 + w1]
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1 - h1) / 2), int((w1 - h1) / 2), 0, 0,
                                                      cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1 - w1) / 2), int((h1 - w1) / 2),
                                                      cv2.BORDER_CONSTANT, (0, 0, 0))
                save_img = cv2.resize(save_img, (image_x, image_y))
                cv2.putText(frame, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
                cv2.imwrite("../100daysofMLcode/DataSet/Gestures/" + str(g_id) + "/" + str(pic_no) + ".jpg", save_img)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.imshow("Capturing gesture", frame)
        cv2.imshow("thresh", thresh)
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            if flag_start_capturing == False:
                flag_start_capturing = True
            else:
                flag_start_capturing = False
                frames = 0
        if flag_start_capturing == True:
            frames += 1
        if pic_no == total_pics:
            break

g_id = input('Enter Gesture Id ')
strore_images(g_id)
