import cv2
import numpy as np

title1 = 'test_zeros_image'
title2 = 'test_ones_image'
title3 = 'test_randint_image'

# 200 x 400 모든 원소가 0인 넘파이 배열 선언
image1 = np.zeros((200, 400), np.uint8)
image2 = np.ones((300, 300), np.uint8)
image3 = np.random.randint(0, 256, size=(400, 500, 3))
image3 = np.array(image3, dtype=np.uint8)

# 윈도우 창 이름 및 크기 조정 옵션(윈도우 창을 이동 하기 위해서는 해당 작업을 해줘야함)
cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE)    # 크기 변경 불가능
cv2.namedWindow(title2, cv2.WINDOW_NORMAL)      # 크기 변경 가능

# 윈도우 창 이동
cv2.moveWindow(title1, 150, 200)
cv2.moveWindow(title2, 300, 300)

# 윈도우 출럭(제목은 title, 영상은 image)
cv2.imshow(title1, image1)
cv2.imshow(title2, image2)
cv2.imshow(title3, image3)

# 윈도우 창 크기 변경(WINDOW_NORMAL 일 시 변경 가능)
cv2.resizeWindow(title1, 500, 500)
cv2.resizeWindow(title2, 500, 500)

# 키 입력 대기
cv2.waitKey()

# 윈도우 종료
cv2.destoryAllWindows()