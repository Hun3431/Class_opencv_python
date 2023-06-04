import numpy as np
import cv2

def color_candidate_img(image, center):
    h, w = image.shape[:2]
    fill = np.zeros((h + 2, w + 2), np.uint8)
    dif1, dif2 = (25, 25, 25), (25, 25, 25)
    flags = 0xff00 + 4 + cv2.FLOODFILL_FIXED_RANGE
    flags += cv2.FLOODFILL_MASK_ONLY

    pts = np.random.randint(-15, 15, (20, 2))
    pts = pts + center
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            _, _, fill, _ = cv2.floodFill(image, fill, (x, y), 255, dif1, dif2, flags)

    return cv2.threshold(fill, 120, 255, cv2.THRESH_BINARY)[1]
def rotate_plate(image, rect):
    center, (w, h), angle = rect

    crop_img = cv2.getRectSubPix(image, (w, h), center)
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    return cv2.resize(crop_img, (144, 28))