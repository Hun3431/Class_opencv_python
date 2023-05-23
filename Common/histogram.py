import cv2
import numpy as np

def make_palette(rows):
    hue = [round(i * 180 / rows) for i in range(rows)]
    hsv = [[[h, 255, 255]] for h in hue]
    hsv = np.array(hsv, np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)