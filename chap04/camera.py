import cv2

capture = cv2.VideoCapture(1)       # 0번 카메라 연결
if capture.isOpened() == False:     # 카메라 연결 안된 경우 예외처리
    raise Exception("카메라 연결 안됨")

title = "View Frame from Camera"    # 출력 창 title
while True:
    ret, frame = capture.read()     # 카메라 영상 받기, 정상적으로 frame(영상) 가져 오면 ret 는 true
    if not ret: break               # frame 못 받으면 종료
    frame[100:300, 200:300, 1] = frame[100:300, 200:300, 1] + 50
    cv2.rectangle(frame, (200, 100), (300, 300), (0, 0, 255), 3)
    cv2.imshow(title, frame)        # 윈도우에 카메라 입력 영상 출력
    if cv2.waitKey(30) >= 0: break  # 키보드 누르면 종료, 30 frame/sec
capture.release()                   # 카메라 연결 해제