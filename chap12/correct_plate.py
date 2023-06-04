from header.plate_preprocess import *
from header.plate_candidate import *

car_no = 0
while True:
    cv2.destroyAllWindows()
    image, morph = preprocessing(car_no)
    candidates = find_candidates(morph)

    cv2.imshow("test", morph)
    cv2.waitKey()

    fills = [color_candidate_img(image, size) for size, _, _ in candidates]
    new_candis = [find_candidates(fill) for fill in fills]
    new_candis = [cand[0] for cand in new_candis if cand]
    candidate_imgs = [rotate_plate(image, cand) for cand in new_candis]

    svm = cv2.ml.SVM_load("./header/SVMtrain.xml")
    rows = np.reshape(candidate_imgs, (len(candidate_imgs), -1))
    _, results = svm.predict(rows.astype('float32'))
    correct = np.where(results == 1)[0]

    print(f"분류 결과 : {results}")
    print(f"번호판 영상 인덱스 : {correct}")

    for i, idx in enumerate(correct):
        cv2.imshow(f"plat image_{str(i)}", candidate_imgs[idx])
        cv2.resizeWindow(f"plat image_{str(i)}", (250, 28))

    for i, candi in enumerate(new_candis):
        color = (0, 255, 0) if i in correct else (0, 0, 255)
        cv2.polylines(image, [np.int32(cv2.boxPoints(new_candis[i]))], True, color, 2)

    print("번호판 검출 완료") if len(correct) > 0 else print("번호판 미검출")
    # for i, img in enumerate(candidate_imgs):
    #     cv2.polylines(image, [np.int32(cv2.boxPoints(new_candis[i]))], True, (0, 255, 255), 2)
    #     cv2.imshow(f"candidate_img - {i}", img)

    title = f"car-{car_no:02d}"
    cv2.namedWindow(title)
    cv2.moveWindow(title, 0, 100)
    cv2.imshow(title, image)

    key = cv2.waitKey()
    print(key)
    if key == 2 and car_no != 0:
        car_no -= 1
    elif key == 3 and car_no != 14:
        car_no += 1
    elif key == 27:
        break
