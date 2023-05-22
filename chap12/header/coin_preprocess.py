import cv2
import numpy as np


def preprocessing(coin_no):
    frame = "images/coin/{0:02d}.png".format(coin_no)
    image = cv2.imread(frame, cv2.IMREAD_COLOR)
    if image is None: return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2, 2)
    flag = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    _, th_img = cv2.threshold(gray, 130, 255, flag)

    mask = np.ones((3, 3), np.uint8)
    th_img = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, mask)
    return image, th_img

def find_coins(image):
    result = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0] if int(cv2.__version__[0]) >= 4 else result[1]

    circles = [cv2.minEnclosingCircle(c) for c in contours]
    circles = [(tuple(map(int, center)), int(radius)) for center, radius in circles if radius > 25]

    return circles

def make_coin_img(src, circles):
    coins = []
    for center, radius in circles:
        r = radius * 3
        cen = (r // 2, r // 2)
        mask = np.zeros((r, r, 3), np.uint8)
        cv2.circle(mask, cen, radius, (255, 255, 255), cv2.FILLED)

        coin = cv2.getRectSubPix(src, (r, r), center)
        coin = cv2.bitwise_and(coin, mask)
        coins.append(coin)
    return coins