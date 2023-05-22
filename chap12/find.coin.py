import cv2
from header.coin_preprocess import *

imageNum = 0

while True:
    cv2.destroyAllWindows()
    image, binary = preprocessing(imageNum)
    if image is None: raise Exception("None Image")

    count = 0

    circles = find_coins(binary)

    coin_no = 0
    coin_img = make_coin_img(image, circles)

    for img in coin_img:
        title = str(coin_no) + "img"
        cv2.namedWindow(title)
        cv2.moveWindow(title, (coin_no % 5) * 150, (coin_no // 5) * 150)
        cv2.imshow(title, img)
        coin_no += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for center, radius in circles:
        count += 1
        text = (center[0] - radius * 1 // 10 - 15, center[1]+10)
        print(count, " - radius : ", radius)
        cv2.circle(image, center, radius, (0, 255, 0), 2)

    cv2.putText(image, (f'./images/coin/{imageNum:02d}.jpg'), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(image, (f'Coin : {count}'), (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(f'Coin_{imageNum:02d}', image)

    # cv2.imshow("Binary", binary)
    Key = cv2.waitKey(0)

    if Key == 27: break;
    elif Key == 2 & Key > 0: imageNum -= 1
    elif Key == 3 & Key < 86: imageNum += 1