import cv2

image_color = cv2.imread("./aa.jpg", cv2.IMREAD_COLOR)

image_color[50:200, 50:200, 1] += 50

cv2.imshow("Color", image_color)

cv2.waitKey()