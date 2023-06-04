from header.plate_preprocess import *
from header.plate_candidate import *

svm = cv2.ml.SVM_load("./header/SVMtrain.xml")
img_no = 0
while True:
    next = False
    image = cv2.imread(f"./images/car/{img_no:02d}.jpg")

    initial_x = 0  # 초기 x 좌표
    initial_y = image.shape[0] // 2  # 초기 y 좌표
    initial_width = 158  # 초기 윈도우의 너비
    initial_height = 30  # 초기 윈도우의 높이
    x_increment = 10  # x 좌표의 증가량
    y_increment = 10  # y 좌표의 증가량
    width_increment = 1.09  # 윈도우 너비의 증가량
    height_increment = 1.09  # 윈도우 높이의 증가량
    max_width = 209  # 최대 윈도우 너비
    max_height = 38  # 최대 윈도우 높이

    x = initial_x
    y = initial_y
    width = initial_width
    height = initial_height

    find = image.copy()

    while width <= max_width and height <= max_height:
        while y <= image.shape[0] - height:
            while x <= image.shape[1] - width:
                test = image.copy()
                window = find[y:y + height, x:x + width].copy()

                # 사각형 좌표
                top_left = (x, y)
                bottom_right = (x + width, y + height)

                # 사각형 그리기
                cv2.rectangle(test, top_left, bottom_right, (0, 255, 0), 2)

                window = cv2.resize(window, (144, 28))

                cv2.imshow(" ", window)

                window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                rows = window.flatten()
                rows = np.array([rows], )

                _, results = svm.predict(rows.astype('float32'))
                correct = np.where(results == 1)[0]

                if correct.size > 0:
                    print("번호판이 감지되었습니다.")
                    cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
                    cv2.imshow(f"{img_no:02d}.jpg", test)
                    # key = cv2.waitKey(1)

                else:
                    cv2.imshow(f"{img_no:02d}.jpg", test)
                    # key = cv2.waitKey(1)

                # if key == 32:
                #     next = True
                #     break

                x += x_increment
            # if (next): break
            x = initial_x
            y += y_increment
        # if (next): break
        x = initial_x
        y = initial_y
        width = int(width * width_increment)
        height = int(height * height_increment)


    cv2.imshow(f"{img_no:02d}.jpg", image)
    #cv2.waitKey(0)
    print(f"이미지 저장 : {img_no:02d}.jpg")
    cv2.imwrite(f"./images/find_image/{img_no:02d}.jpg", image)
    img_no += 1
    cv2.waitKey(1)

    if img_no == 15: break


##########################################################################################################33


    # rows = np.reshape(candidate_imgs, (len(candidate_imgs), -1))
    # _, results = svm.predict(rows.astype('float32'))
    # correct = np.where(results == 1)[0]

    #
    # row_image = np.reshape(test_image, (len(test_image), -1)).astype('float32')
    # _, images = svm.predict(row_image)
    #
    # correct = np.where(images == 1)[0]
    #
    # for img in images:
    #     cv2.imshow("title", img)
    #     cv2.waitKey()
    #
    #
    # cv2.destroyAllWindows()
    # image, morph = preprocessing(car_no)
    # candidates = find_candidates(morph)
    #
    # cv2.imshow("test", morph)
    # cv2.waitKey()
    #
    # fills = [color_candidate_img(image, size) for size, _, _ in candidates]
    # new_candis = [find_candidates(fill) for fill in fills]
    # new_candis = [cand[0] for cand in new_candis if cand]
    # candidate_imgs = [rotate_plate(image, cand) for cand in new_candis]
    #
    #
    # # 이부분 고정
    # svm = cv2.ml.SVM_load("./header/SVMtrain.xml")
    # rows = np.reshape(candidate_imgs, (len(candidate_imgs), -1))
    # _, results = svm.predict(rows.astype('float32'))
    # correct = np.where(results == 1)[0]
    #
    # print(f"분류 결과 : {results}")
    # print(f"번호판 영상 인덱스 : {correct}")
    #
    # for i, idx in enumerate(correct):
    #     cv2.imshow(f"plat image_{str(i)}", candidate_imgs[idx])
    #     cv2.resizeWindow(f"plat image_{str(i)}", (250, 28))
    #
    # for i, candi in enumerate(new_candis):
    #     color = (0, 255, 0) if i in correct else (0, 0, 255)
    #     cv2.polylines(image, [np.int32(cv2.boxPoints(new_candis[i]))], True, color, 2)
    #
    # print("번호판 검출 완료") if len(correct) > 0 else print("번호판 미검출")
    # # for i, img in enumerate(candidate_imgs):
    # #     cv2.polylines(image, [np.int32(cv2.boxPoints(new_candis[i]))], True, (0, 255, 255), 2)
    # #     cv2.imshow(f"candidate_img - {i}", img)
    #
    # title = f"car-{car_no:02d}"
    # cv2.namedWindow(title)
    # cv2.moveWindow(title, 0, 100)
    # cv2.imshow(title, image)
    #
    # key = cv2.waitKey()
    # print(key)
    # if key == 2 and car_no != 0:
    #     car_no -= 1
    # elif key == 3 and car_no != 14:
    #     car_no += 1
    # elif key == 27:
    #     break


# import cv2
# import numpy as np
#
#
# def sliding_window(image, initial_x, initial_y, initial_width, initial_height, x_increment, y_increment, width_increment, height_increment, max_width, max_height):
#     x = initial_x
#     y = initial_y
#     width = initial_width
#     height = initial_height
#
#     while width <= max_width and height <= max_height:
#         while y <= image.shape[0] - height:
#             while x <= image.shape[1] - width:
#                 test = image.copy()
#                 window = image[y:y + height, x:x + width]
#
#                 # 사각형 좌표
#                 top_left = (x, y)
#                 bottom_right = (x + width, y + height)
#
#                 # 사각형 그리기
#                 cv2.rectangle(test, top_left, bottom_right, (0, 255, 0), 2)
#
#                 rows = np.reshape(window, (len(window), -1)).astype('float32')
#                 _, results = svm.predict(rows)
#                 correct = np.where(results == 1)[0]
#
#                 if correct.size > 0:
#                     # 번호판으로 판별되었을 때의 처리를 수행합니다.
#                     print("번호판이 감지되었습니다.")
#                     # 예를 들면, 번호판 영역을 추출하거나 번호판을 인식하는 등의 작업을 수행할 수 있습니다.
#
#                     # 번호판 영역 표시
#                     cv2.putText(test, 'License Plate', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                     cv2.imshow("test", test)
#                     cv2.waitKey(0)
#                 else:
#                     cv2.imshow("test", test)
#                     cv2.waitKey(50)
#
#                 x += x_increment
#             x = initial_x
#             y += y_increment
#         x = initial_x
#         y = initial_y
#         width += width_increment
#         height += height_increment
#
#
# # SVM 모델 로드
# svm = cv2.ml.SVM_load("./test/SVMtrain.xml")
# image = cv2.imread("./images/car/00.jpg")
#
# initial_x = 0  # 초기 x 좌표
# initial_y = image.shape[0] // 2  # 초기 y 좌표
# initial_width = 100  # 초기 윈도우의 너비
# initial_height = 20  # 초기 윈도우의 높이
# x_increment = 40  # x 좌표의 증가량
# y_increment = 20  # y 좌표의 증가량
# width_increment = 40  # 윈도우 너비의 증가량
# height_increment = 10  # 윈도우 높이의 증가량
# max_width = 300  # 최대 윈도우 너비
# max_height = 60  # 최대 윈도우 높이
#
# # 슬라이딩 윈도우 실행
# sliding_window(image, initial_x, initial_y, initial_width, initial_height, x_increment, y_increment, width_increment, height_increment, max_width, max_height)
