import numpy as np
import cv2

image = np.array([255 for i in range(3000000)], np.uint8).reshape(1000,1000,3)

white = np.zeros((1000, 1000, 3), np.uint8)
white[:] = 255

cv2.line(image, (300, 300), (200, 500), (255, 0, 0), 10)
cv2.rectangle(image, (200, 200), (600, 600), (0, 255, 0), 20)
cv2.putText(image, "Hello OpenCV", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 128, 0));
cv2.circle(image, (500, 500), 100, (0, 0, 0), 5)


cv2.imshow("white", white)

cv2.imshow("title", image)
cv2.waitKey()
