import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)#edge deteced
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200, height), (1100,height), (550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)#line is 2d array of 2x1, convert to many 1d array
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)#image_to_be_drwan,first_co-ordinates,second_co-ordinates,colo_of_line,thickness
    return line_image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]#bootom of image
    y2 = int(y1*(3/5))#line to be displayed upto 3/5 height
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)#degree of polynomial=1
        slope = parameters[0]
        intercept = parameters[1]
        #Lines on left will have -ve slope and line on right will have +ve slope
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

cap = cv2.VideoCapture('../100daysofMLcode/Day 15-/test_video.mp4')
video = cv2.VideoWriter('../100daysofMLcode/Day 15-/detected.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 50, (1280,720), 1)
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_line(frame, averaged_lines)
    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    video.write(combined_image)
    cv2.imshow('Region of interest', combined_image)
    if cv2.waitKey(1) == ord('q'):#1 milliseconds b/w frames
        break
cap.release()
video.release()
cv2.destroyAllWindows()
