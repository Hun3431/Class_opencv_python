import numpy as np
import cv2


def imgTrim(img, x, y, w, h):
    if x > w :
        copy = x
        x = w
        w = copy
    if y > h :
        copy = y
        y = h
        h = copy
    imtrim = img[y+1:h-1, x+1:w-1]
    return imtrim

def onMouse(event, x, y, flags, param):
    global x1, y1
    img = param.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.imshow(title, img)
        x1, y1 = x, y
    elif event == cv2.EVENT_LBUTTONUP:

        cv2.rectangle(img, (x1, y1), (x, y), (0, 255, 0), 2)
        cv2.imshow(title, img)
        result = imgTrim(img, x1, y1, x, y)
        img1 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img2', img1)


img = cv2.imread('aa.jpg').copy()
title = 'img'

cv2.imshow(title, img)
cv2.setMouseCallback(title, onMouse, img)

cv2.waitKey()
cv2.destroyAllWindows()
