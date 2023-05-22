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

sum_image = np.zeros((height, width), dtype=np.float32)

for i in range(len(image_files)):
    sum_image += np.array(image_files[i], dtype=np.float32)

average = sum_image / image_count

cv2.imshow("Average Image", average.astype(np.uint8))

cv2.waitKey()