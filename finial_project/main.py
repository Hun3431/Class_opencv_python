import numpy as np
import cv2
import copy
image_count = 310
width = 120
height = 150
size = width * height

image_files = []

for i in range(image_count):
    image = cv2.imread(f"./face_img/train/train{i:03d}.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (width, height))
    image_files.append(image)

sum_image = np.zeros((height, width), dtype=np.float32)

for i in range(len(image_files)):
    sum_image += np.array(image_files[i], dtype=np.float32)

average = sum_image / image_count

# cv2.imshow("Average Image", average.astype(np.uint8))

difference_array = np.zeros((size, 1), dtype=np.uint8)

for i in range(len(image_files)):
    image = copy.deepcopy(image_files[i])
    image = image - average
    image = image.reshape(size, 1)
    difference_array = np.append(difference_array, image, axis=1)

difference_array = np.delete(difference_array, 0, axis=1)

print(difference_array)

cv2.waitKey()