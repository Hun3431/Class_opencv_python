import numpy as np
import cv2

def preprocessing(car_no):
    image = cv2.imread(f"images/car/{car_no:02d}.jpg", cv2.IMREAD_COLOR)
    if image is None: return None, None

    kernel = np.ones((5, 17), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5, 5))
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)

    th_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
    morph = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel, iterations=3)

    return image, morph

def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: return False

    aspect = h / w if h > w else w / h
    chk1 = 3000 < (h * w) < 12000
    chk2 = 2.0 < aspect < 6.5

    return chk1 and chk2

def find_candidates(image):
    results = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    rects = [cv2.minAreaRect(c) for c in contours]
    candidates = [(tuple(map(int,center)), tuple(map(int, size)), angle) for center, size, angle in rects if verify_aspect_size(size)]

    return candidates