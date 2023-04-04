import cv2
#from Common.utils import print_matInfo

title1, title2 = 'color2gray', 'color2color'
color2unchanged1 = cv2.imread("../face/16.jpg", cv2.IMREAD_UNCHANGED)
color2unchanged2 = cv2.imread("../face/32.jpg", cv2.IMREAD_UNCHANGED)

if color2unchanged1 is None or color2unchanged2 is None:
    raise Exception("영상파일 읽기 에러")

cv2.imshow(title1, color2unchanged2)
cv2.imshow(title2, (color2unchanged2 * 255).astype('uint8'))
cv2.waitKey(0)