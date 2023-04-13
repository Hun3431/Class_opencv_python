import cv2

face_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_mcs_mouth.xml")

def preprocessing(image):
    if image is None: return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return image, gray

def find_face(gray):
    return face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))

def find_eye(gray):
    return eye_cascade.detectMultiScale(gray, 1.05, 7, 0, (20, 20))

def find_mouth(gray):
    return mouth_cascade.detectMultiScale(gray, 1.2, 4, 0, (23, 10))

print("1. 사진에서 얼굴 찾기")
print("2. 카메라에서 얼굴 찾기")

num = input("-> ")

if int(num) == 1:
    print("1. 사진에서 얼굴 찾기")
    print("- 파일 번호를 입력하시오.")
    print("( 0 ~ 64 )")
    image_num = int(input("-> "))

    if image_num < 0 | image_num > 64:
        print("파일 번호 입력이 잘못되었습니다.")

    else:
        print("종료하려면 ESC를 눌러주세요")
        print("방향키로 조절이 가능합니다.")
        print(" <-  -> ")
        while True:
            image = cv2.imread('images/face/%02d.jpg' %image_num, cv2.IMREAD_COLOR)
            image, gray = preprocessing(image)

            if image is None:
                raise Exception("영상 파일 읽기 에러")

            title = "image" + str(image_num)

            faces = find_face(gray)
            if len(faces):
                print("얼굴", len(faces), "개 검출")
                for face in faces:
                    x, y, w, h = face
                    face_up = gray[y:y + h // 2, x:x + w]
                    eyes = find_eye(face_up)
                    if len(eyes) > 1:
                        for ex, ey, ew, eh in eyes:
                            center = (x + ex + ew // 2, y + ey + eh // 2)
                            cv2.circle(image, center, 10, (0, 255, 0), 2)
                    else:
                        print("눈 미검출")
                    face_down = gray[y + (h // 3) * 2:y + h, x:x + w]

                    mouths = find_mouth(face_down)
                    if len(mouths) > 0:
                        for mx, my, mw, mh in mouths:
                            if mw * 4 < w:
                                continue
                            cv2.rectangle(image, (x + mx, y + (h // 3) * 2 + my), (x + mx + mw, y + (h // 3) * 2 + my + mh), (0, 0, 255), 2)
                    else:
                        print("입 미검출")

                    cv2.rectangle(image, face, (255, 0, 0), 2)

            cv2.imshow(title, image)
            key = cv2.waitKey()
            if key == 2:
                if image_num != 0:
                    image_num -= 1
            elif key == 3:
                if image_num != 64:
                    image_num += 1
            elif key == 27:
                break
            cv2.destroyAllWindows()

elif int(num) == 2:
    print("2. 카메라에서 얼굴 찾기")
    title = "View Frame from Camera"

    capture = cv2.VideoCapture(0)  # 0번 카메라 연결
    if capture.isOpened() == False:  # 카메라 연결 안된 경우 예외처리
        raise Exception("카메라 연결 안됨")

    while True:
        ret, frame = capture.read()  # 카메라 영상 받기, 정상적으로 frame(영상) 가져 오면 ret 는 true
        if not ret: break  # frame 못 받으면 종료

        image, gray = preprocessing(frame)
        faces = find_face(gray)

        if len(faces) > 0:
            print("얼굴", len(faces), "개 검출")
            # 검출된 얼굴에 대해서 반복
            for face in faces:
                # 얼굴 영역 추출
                x, y, w, h = face

                face_up = gray[y:y + h // 2, x:x + w]
                # 눈 검출
                eyes = find_eye(face_up)
                # 눈이 두개인 경우
                if len(eyes) == 2:
                    for ex, ey, ew, eh in eyes:
                        center = (x + ex + ew // 2, y + ey + eh // 2)
                        cv2.circle(image, center, 10, (0, 255, 0), 2)
                else:
                    print("눈 미검출")

                cv2.rectangle(image, face, (255, 0, 0), 2)

                face_down = gray[y + (h // 3) * 2:y + h, x:x + w]

                mouths = mouth_cascade.detectMultiScale(face_down, 1.3, 6, 0, (40, 20))
                if len(mouths) > 0:
                    for mx, my, mw, mh in mouths:
                        if mw * 4 < w:
                            continue
                        cv2.rectangle(image, (x + mx, y + (h // 3) * 2 + my), (x + mx + mw, y + (h // 3) * 2 + my + mh), (0, 0, 255), 2)
                else:
                    print("입 미검출")
            # cv2.imshow(title + str(image_num), image)

        else:
            print("얼굴 미검출")
            # cv2.imshow(title + str(image_num), image)

        cv2.imshow(title, frame)  # 윈도우에 카메라 입력 영상 출력

        if cv2.waitKey(10) >= 0: break  # 키보드 누르면 종료, 30 frame/sec
    capture.release()
