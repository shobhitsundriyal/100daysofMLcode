import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
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

image = cv2.imread('../100daysofMLcode/Day 15-/test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)
###############################pixel_val,radian,treshold_for_bins,empty_array,
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
line_image = display_line(lane_image, lines)
combined_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#cv2.imshow('result', canny)
#cv2.waitKey(0)
cv2.imshow('Region of interest', combined_image)
cv2.waitKey(0)
#cv2.imshow('result', gray)
#cv2.waitKey(0)
