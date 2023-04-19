import cv2
import numpy as np

image = np.array([i % 256 for i in range(200000)], np.uint8).reshape(400, 500)
cv2.imshow("title", image)

while True:
    # Key 입력 대기
    key = cv2.waitKey(30)
    # ESC 입력 시 종료
    if key == 27: break
    # 입력 받은 Key 출력
    elif key != -1: print(chr(key))
cv2.waitKey()
cv2.destroyAllWindows()