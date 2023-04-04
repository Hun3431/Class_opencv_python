import cv2, numpy as np
import requests

def proprocessing_by_web():
    image_nparray = np.asarray(bytearray(requests.get("https://source.unsplash.com/random/?person").content), dtype=np.uint8)
    image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)

    # image = cv2.imread('./images/face/%02d.jpg' % no, cv2.IMREAD_COLOR)
    if image is None: return None,None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return image, gray

image_file = 6

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    image, gray = proprocessing_by_web()
    if image is None: raise Exception("영상파일 읽기 에러")

    faces = face_cascade.detectMultiScale(gray, 1.1, 2, 0, (100, 100))

    if len(faces) > 0:
        for face in faces:
            x, y, w, h = face
            face_image = image[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_image, 1.15, 7, 0, (25, 20))

            if len(eyes) == 2:
                for (ex, ey, ew, eh) in eyes:
                    center = (x + ex + ew // 2, y + ey + eh // 2)
                    cv2.circle(image, center, 10, (0, 255, 0), 2)
            else:
                print("눈 미검출")

            cv2.rectangle(image, face, (255, 0, 0), 2)
            cv2.imshow("image", image)

    else:
        print("얼굴 미검출")
        cv2.imshow("image", image)
    cv2.waitKey()