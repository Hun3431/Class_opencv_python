import numpy as np
import cv2

def onMouse(event, x, y, flags, param):
    global pt1, pt2, drag

    if event == cv2.EVENT_LBUTTONDOWN:
        drag = True
        pt1 = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag == True:
            pt2 = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drag = False
        pt2 = (x, y)
img = cv2.imread('../chap04/aa.jpg')
title = 'img'

cv2.imshow(title, img)
cv2.setMouseCallback(title, onMouse)

drag = False
pt1 = (0, 0)
pt2 = (0, 0)

while True:
    temp_img = img.copy()

    if drag == True:
        cv2. rectangle(temp_img, pt1, pt2, (0, 255, 0), 1)

    cv2.imshow(title, temp_img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.waitKey()
cv2.destroyAllWindows()