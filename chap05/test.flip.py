import numpy as np
import cv2

image = np.array([i%256 for i in range(1000000)], np.uint8).reshape(1000,1000)

cv2.circle(image, (300, 300), 100, (255, 0, 0), 10)\

image2 = cv2.flip(image, 0)
image3 = cv2.flip(image, 1)
image4 = cv2.flip(image, -1)

cv2.imshow("title", image)      # 원본
cv2.imshow("flipx", image2)     # 상하 반전
cv2.imshow("flipy", image3)     # 좌우 반전
cv2.imshow("flip-1", image4)    # 둘 다 반전
cv2.waitKey()
