import cv2
import numpy as np

prototxt = "./SSD_data/deploy.prototxt" #
caffemodel = "./SSD_data/res10_300x300_ssd_iter_140000_fp16.caffemodel"     # 학습 모델
detector = cv2.dnn.readNet(prototxt, caffemodel);

num = 0

def show (image, state):
    global num
    #copyimage = image.copy
    (h, w) = image.shape[:2]
    target_size = (300, 300)
    input_image = cv2.resize(image, target_size)
    imageBlob = cv2.dnn.blobFromImage(input_image)
    detector.setInput(imageBlob)
    detections = detector.forward()

    results = detections[0][0]
    threshold = 0.8

    arr = []

    for i in range(0, results.shape[0]):
        conf = results[i, 2]
        if conf < threshold:
            continue
        box = results[i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype('int')
        if state:
            print("CV_11_3_20193210/1_20193210_{:1d}.jpg 저장".format(num))
            cv2.imwrite("./CV_11_3_20193210/1_20193210_{:02d}.jpg".format(num), image[startY:endY, startX:endX])
            num += 1

        cv2.putText(image, str(conf), (startX, startY - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)  # 얼굴위치 box 그리기
    return image, arr


#image = cv2.imread("./images/face/07.jpg")
title = "View Frame from Camera"

capture = cv2.VideoCapture(0)  # 0번 카메라 연결
if capture.isOpened() == False:  # 카메라 연결 안된 경우 예외처리
    capture = cv2.VideoCapture(1)  # 1번 카메라 연결
    if capture.isOpened() == False:  # 예외처리
        raise Exception("카메라 연결 안됨")

state = False

while True:
    ret, frame = capture.read()  # 카메라 영상 받기, 정상적으로 frame(영상) 가져 오면 ret 는 true
    if not ret: break  # frame 못 받으면 종료
    image, images = show(frame, state)   # 카메라에서 받은 영상에서 얼굴 인식
    state = False
    cv2.imshow('image', image)   # 얼굴을 찾은 이미지 파일을 출력
    key = cv2.waitKey(13)
    if key == 27: break  # 키보드 누르면 종료, 30 frame/sec
    if key == 32:
        state = True
        print("화면 캡쳐")

capture.release()