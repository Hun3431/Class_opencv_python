import numpy as np
import cv2

image_count = 310
width = 120
height = 150

image_files = []

for i in range(image_count):
    image = cv2.imread(f"./face_img/train/train{i:03d}.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (width, height))
    image_files.append(image)