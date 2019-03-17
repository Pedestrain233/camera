from collections import deque
import numpy as np
import cv2
import time
import TIS

redLower = np.array([170,100,100])
redUpper = np.array([179,255,255])

mybuffer = 64
pts =deque(maxlen=mybuffer)

Tis = TIS.TIS("35814519",640,480,120,True)

Tis.Start_pipeline()

print('Press Esc to stop')
lastkey = 0

cv2.namedWindow('Window')

while lastkey != 27:
 if Tis.Snap_image(1) is True:
     image = Tis.Get_image()
     cv2.imshow('Window', image)
     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
     mask = cv2.inRange(hsv, redLower, redUpper)
     mask = cv2.erode(mask, None, iterations=2)
     mask = cv2.dilate(mask, None, iterations=2)
     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
     center = None
     if len(cnts) > 0:
      c = max(cnts, key = cv2.contourArea)
     ((x, y), radius) = cv2.minEnclosingCircle(c)
     M = cv2.moments(c)
     center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
     if radius > 10:
      cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
     cv2.circle(frame, center, 5, (0, 0, 255), -1)
     pts.appendleft(center)
     for i in xrange(1, len(pts)):
      if pts[i - 1] is None or pts[i] is None:
       continue
     thickness = int(np.sqrt(mybuffer / float(i + 1)) * 2.5)
     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
     cv2.imshow('Frame', frame)
     lastkey = cv2.waitKey(10)

Tis.Stop_pipeline()
cv2.destroyAllWindows()
print('Program ends')
