import cv2
import matplotlib.pyplot as plt

image = cv2.imread("./face/00.jpg", cv2.IMREAD_COLOR)
if image is None: raise Exception("영상파일 읽기 에러")

rows, cols = image.shape[:2]
rgb_img = cv2.cvtColor(image, cv2.COLOR_RGB2RGB)
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

fig = plt.figure(num=1, figsize=(3,4))
plt.imshow(image), plt.title('title-figure')
plt.axis('off'), plt.tight_layout()

fig = plt.figure(figsize=(6,4))
plt.suptitle('title-figure-new')
plt.subplot(1,2,1), plt.imshow(rgb_img)
plt.axis([0, cols, rows, 0]), plt.title('rgb-color')
plt.subplot(1,2,2), plt.imshow(gray_img, camp='gray')
plt.title('gray_img2')
plt.show()