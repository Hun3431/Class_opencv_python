import cv2, numpy as np

image_num = 1

def preprocessing(no):
    image = cv2.imread('images/face/%02d.jpg' %no, cv2.IMREAD_COLOR)
    if image is None: return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return image, gray

face_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_frontalface_alt2.xml")
eye_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier("images/Haar_files/haarcascade_mcs_mouth.xml")

title = "img_"
title = "img_"

while(image_num < 68):

    #이미지 불러오기
    image, gray = preprocessing(image_num)
    if image is None:
        raise Exception("영상 파일 읽기 에러")

    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))

    # 얼굴을 찾았을 경우 실행
    if len(faces):
        # 검출된 얼굴에 대해서 반복
        for face in faces:
            print("face : ", face)
            # 얼굴 영역 추출
            x,y,w,h = face

            face_up = image[y:y+h//2, x:x+w]
            cv2.rectangle(image, (x, y), (x+w, y+h//2), (0, 0, 255), 1)
            # 눈 검출
            eyes = eye_cascade.detectMultiScale(face_up, 1.2, 4, 0, (23, 20))
            # 눈이 두개인 경우
            if len(eyes):
                for ex, ey, ew, eh in eyes:
                    center = (x + ex + ew // 2, y + ey + eh // 2)
                    cv2.circle(image, center, 10, (0,255,0), 2)
            else:
                print("눈 미검출")

            cv2.rectangle(image, face, (255, 0, 0), 2)

            face_down = image[y+(h//3)*2:y+h, x:x+w]

            mouth = mouth_cascade.detectMultiScale(face_down, 1.1, 3, 0, (20,20))
            if len(mouth):
                for mx, my, mw, mh in mouth:
                    print("mouth : ", mouth)
                    if mw * 4 < w:
                        continue
                    cv2.rectangle(image, (x + mx, y+(h//3)*2 + my), (x + mx + mw, y+(h//3)*2 + my + mh), (0, 0, 255), 2)
                    break
            else:
                print("입 미검출")

        cv2.imshow(title + str(image_num), image)

    else:
        print("얼굴 미검출")
        cv2.imshow(title + str(image_num), image)
    if(cv2.waitKey() == 27):
        break
    cv2.destroyAllWindows()
    image_num = image_num+1