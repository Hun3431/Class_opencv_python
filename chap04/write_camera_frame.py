import cv2

capture = cv2.VideoCapture(0)       # 0번 카메라 연결
if capture.isOpened() == False:     # 카메라 연결 안된 경우 예외처리
    raise Exception("카메라 연결 안됨")

fps = 29.97
delay = round(1000/fps)
size = (1920, 1080)
fourcc = cv2.VideoWriter_fourcc(*'DX50')

capture.set(cv2.CAP_PROP_FOCUS, 0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

title = "View Frame from Camera"    # 출력 창 title

writer = cv2.VideoWriter("./video.avi", fourcc, fps, size)
if writer.isOpened() == False: raise Exception("동영상 개방 안됨")

while True:
    ret, frame = capture.read()     # 카메라 영상 받기, 정상적으로 frame(영상) 가져 오면 ret 는 true
    if not ret: break               # frame 못 받으면 종료
    if cv2.waitKey(30) >= 0: break  # 키보드 누르면 종료, 30 frame/sec

    writer.write(frame)
    cv2.imshow(title, frame)
writer.release()
capture.release()                   # 카메라 연결 해제