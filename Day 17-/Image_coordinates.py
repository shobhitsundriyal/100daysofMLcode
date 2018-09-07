import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib
import requests

url = 'http://192.168.1.3:8080/shot.jpg'
frame_url = urllib.request.urlopen(url)
frame_np = np.array(bytearray(frame_url.read()))
frame_np
frame = cv2.imdecode(frame_np, -1)
frame
plt.imshow(frame)
plt.show()
