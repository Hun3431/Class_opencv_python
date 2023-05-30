from header.plate_preprocess import *

car_no = int(input("자동차 영상번호(0 ! 15): "))
image, morph = preprocessing(car_no)
if image is None: Exception("영상파일 읽기 에러")

candidates = find_candidates(morph)
for candidate in candidates:
    pts = np.int32(cv2.boxPoints(candidate))
    cv2.polylines(image, [pts], True, (0, 255, 255), 2)
    print(candidate)

if not candidate:
    print("번호판 후보 영역 미검출")
cv2.imshow("image", image)
cv2.waitKey(0)