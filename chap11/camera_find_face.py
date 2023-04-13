import cv2

def preprocessing(image):
    if image is None: return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return image, gray

capture = cv2.VideoCapture(0)       # 0번 카메라 연결
if capture.isOpened() == False:     # 카메라 연결 안된 경우 예외처리
    raise Exception("카메라 연결 안됨")


face_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_mcs_mouth.xml")


title = "View Frame from Camera"    # 출력 창 title
while True:
    ret, frame = capture.read()     # 카메라 영상 받기, 정상적으로 frame(영상) 가져 오면 ret 는 true
    if not ret: break               # frame 못 받으면 종료
    # frame[100:300, 200:300, 1] = frame[100:300, 200:300, 1] + 50
    # cv2.rectangle(frame, (200, 100), (300, 300), (0, 0, 255), 3)

    image, gray = preprocessing(frame)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3, 0, (100, 100))

    # 얼굴을 찾았을 경우 실행
    if len(faces) > 0:
        # 검출된 얼굴에 대해서 반복
        for face in faces:
            # 얼굴 영역 추출
            x, y, w, h = face

            face_up = image[y:y + h // 2, x:x + w]
            cv2.rectangle(image, (x, y), (x + w, y + h // 2), (0, 0, 255), 1)
            # 눈 검출
            eyes = eye_cascade.detectMultiScale(face_up, 1.2, 4, 0, (23, 20))
            # 눈이 두개인 경우
            if len(eyes):
                for ex, ey, ew, eh in eyes:
                    center = (x + ex + ew // 2, y + ey + eh // 2)
                    cv2.circle(image, center, 10, (0, 255, 0), 2)
            else:
                print("눈 미검출")

            cv2.rectangle(image, face, (255, 0, 0), 2)

            face_down = image[y + (h // 3) * 2:y + h, x:x + w]

            mouth = mouth_cascade.detectMultiScale(face_down, 1.3, 6, 0, (40, 20))
            if len(mouth):
                for mx, my, mw, mh in mouth:
                    cv2.rectangle(image, (x + mx, y + (h // 3) * 2 + my), (x + mx + mw, y + (h // 3) * 2 + my + mh),
                                  (0, 0, 255), 2)
                    break
            else:
                print("입 미검출")
        #cv2.imshow(title + str(image_num), image)

    else:
        print("얼굴 미검출")
        #cv2.imshow(title + str(image_num), image)

    cv2.imshow(title, frame)        # 윈도우에 카메라 입력 영상 출력

    if cv2.waitKey(10) >= 0: break  # 키보드 누르면 종료, 30 frame/sec
capture.release()                   # 카메라 연결 해제