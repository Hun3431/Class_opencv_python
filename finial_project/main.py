import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt

# 평균 영상 구하기
def Average_Image(images):
    # resize된 영상의 리스트만을 전달받기 때문에 해당 영상의 가로, 세로 사이즈를 변수에 저장해줌.
    width = images[0].shape[1]
    height = images[0].shape[0]
    # sum
    sum = np.zeros((height, width), dtype=np.float32)
    for i in range(len(images)):
        # 영상을 14개씩 6줄을 출력하고 난 뒤에 모든 영상을 지워줌
        if (i // 14) % 6 == 0 and i % 14 == 0 and i != 0:
            cv2.waitKey(1)
            cv2.destroyAllWindows()

        # 평균 영상을 구하기 위해 sum numpy 배열에 값을 더해줌.
        sum += np.array(images[i], dtype=np.float32)

        # 영상이 정상적으로 열리고 저장이 되고 있는지를 확인하기 위해 모든 영상들을 영상번호에 맞춰서 출력을 반복해줌
        title = f"train{i:03d}.jpg"
        # 이미지들이 겹치지 않도록 위치를 지정해줌
        cv2.namedWindow(title)
        cv2.moveWindow(title, (i % 14) * width, (((i // 14) % 6) * (height + 20)))
        cv2.imshow(title, images[i].astype(np.uint8))

    # 모든 영상을 출력 및 모든 영상의 합을 구한 뒤 출력된 화면들을 모두 제거해줌
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    # 이미지 리스트 파일의 길이를 모든 영상의 합인 sum numpy 배열을 나누어서 평균영상을 구해줌
    return sum / len(images)


# 차영상 구하기
def Difference_Image(images, average):
    # 모든 학습 영상들의 차영상을 구하기 위해 학습 영상 리스트와 평균영상을 전달받음.
    # 평균 영상을 통해 총 화소의 갯수를 계산해줌
    size = average.shape[0] * average.shape[1]
    # 차영상을 담을 numpy 배열을 생성해줌.
    array_image = np.zeros((size, 1), dtype=np.float32)
    for i in range(len(images)):
        # 리스트에서 영상을 꺼내와 해당 영상을 평균 영상으로 빼준 뒤 세로영상으로 변환 해주고 numpy 배열에 추가해줌.
        image = images[i]
        image = image - average
        image = image.reshape(size, 1)
        array_image = np.append(array_image, image, axis=1)

    # numpy 배열 첫 열은 zero값이 들어가 있기 때문에 해당 줄을 지워준 뒤 값을 반환해줌
    return np.delete(array_image, 0, axis=1)


# 고유값/고유벡터 정렬
def Eigen_Sort(value, vector):
    # 고유값을 내림차순으로 정렬한 값을 index 변수에 저장한다.
    index = value.argsort()[::-1]
    # 해당 index를 이용하여 고유값을 정렬하고
    value_sort = value[index]
    # 마찬가지로 index를 이용하여 고유벡터도 정렬한다
    vector_sort = vector[:, index]
    # 정렬 한 고유값과 고유벡터를 반환해준다.
    return value_sort, vector_sort


# 고유값의 주성분 비율을 통해 사용 할 주성분의 갯수를 구하는 함수
def Select_Vector(sort, rate):
    # 정렬 된 고유값에서 실제로 사용할 주성분 만을 가지고 오는 작업
    sum = 0
    # 모든 고유값의 합에 사용할 주성분의 비율인 rate를 곱해주어 기준을 정해줌
    sum_eigen_value = sort.sum() * rate
    for i in range(len(sort)):
        # 고유값의 첫 번째 값 부터 더해가면서 기준을 넘어가거나 같아질 때 해당 Index에 +1을 한 값을 반환해줌
        sum += sort[i]
        if sum_eigen_value <= sum:
            return i + 1


# 변환행렬 축소 구하기
def Transform_Matrix_Reduce(difference_array, eigen_vector_sort, select_index, size):
    # 축소한 변환행렬을 저장해줄 numpy 배열을 선언해줌
    transform_matrix = np.zeros((size, 1))
    # 이전에 구한 주성분의 갯수만큼을 반복해서 돌려줌(0 ~ select_index)
    for i in range(select_index):
        # 차 영상과 고유벡터를 행렬곱해주고, 세로 영상으로 변환해줌
        mul = (difference_array @ eigen_vector_sort[:, i]).reshape(size, 1)
        # 변환된 영상을 정규화 한 뒤 배열에 추가해줌
        transform_matrix = np.append(transform_matrix, mul / np.linalg.norm(mul), axis=1)
    # 처음에 numpy 배열을 생성할 때 만든 0번째 열을 제거한 뒤 반환해줌
    return np.delete(transform_matrix, 0, axis=1)


# 주성분 데이터 투영 구하기
def Calculate_PCA(image_count, difference_array, select_index, transform_matrix):
    # PCA 결과를 저장할 numpy 배열을 선언해줌
    pca_array = np.zeros((select_index, 1))
    for i in range(image_count):
        # 축소한 변환 행렬의 전치를 한 배열과 차이벼열을 행렬곱을 통해여 PCA를 계산하고, PCA numpy배열에 저장합니다.
        pca_array = np.append(pca_array, (transform_matrix.T @ difference_array[:, i]).reshape(select_index, 1), axis=1)

    # 처음에 numpy 배열을 생성할 때 만든 0번째 열을 제거한 뒤에 PCA 결과값을 반환해줌
    return np.delete(pca_array, 0, axis=1)


# PCA 구현 함수
def myPCA(image_files, size, rate):
    image_count = len(image_files)
    # 평균 영상 구하기
    average = Average_Image(copy.deepcopy(image_files))

    cv2.destroyAllWindows()

    cv2.imwrite('Average_Image.jpg', average)
    cv2.imshow("Average Image", average.astype(np.uint8))

    # 차영상 한줄영상
    print("차 영상 구하기 시작")
    difference_array = Difference_Image(copy.deepcopy(image_files), average)
    print("차 영상 구하기 완료")
    print(difference_array)

    # 공분산 행렬
    print("공분산 행렬 구하기 시작")
    covariance_array = np.cov(difference_array.T)
    print("공분산 행렬 구하기 완료")
    print(covariance_array)

    # 고유값 고유벡터
    print("고유값, 고유벡터 구하기 시작")
    eigen_value, eigen_vector = np.linalg.eig(covariance_array)
    print("고유값, 고유벡터 구하기 완료")
    print(eigen_value)
    print(eigen_vector)

    # 고유값/고유벡터 정렬
    print("고유값/고유벡터 정렬 시작")
    eigen_value_sort, eigen_vector_sort = Eigen_Sort(eigen_value, eigen_vector)
    print("고유값/고유벡터 정렬 완료")
    print(eigen_value_sort)
    print(eigen_vector_sort)

    # 고유값 그래프
    plt.title('Eigen Value')
    plt.plot(range(len(eigen_value_sort)), eigen_value_sort)
    plt.show()
    plt.savefig('graph.png')

    # 고유벡터 선택
    print("고유벡터 선택 시작")
    select_index = Select_Vector(eigen_value_sort, rate)
    print("고유벡터 선택 완료")
    print(select_index)

    # 변환 행렬 축소
    print("변환 행렬 축소 시작")
    transform_matrix = Transform_Matrix_Reduce(difference_array, eigen_vector_sort, select_index, size)
    print("변환 행렬 축소 종료")
    print(transform_matrix)
    print(transform_matrix.sum())

    # PCA 배열 구하기
    print("주성분 데이터 투영 시작")
    pca_array = Calculate_PCA(image_count, difference_array, select_index, transform_matrix)
    print("주성분 데이터 투영 종료")
    print(pca_array)

    return average, difference_array, select_index, transform_matrix, pca_array


def main():
    # PCA에 필요한 변수 선언
    image_count = 310   # 학습 영상의 갯수를 나타내는 변수
    width = 120         # 영상의 가로 사이즈를 결정하는 변수
    height = 150        # 영상의 세로 사이즈를 결정하는 변수
    rate = 0.95         # 주성분의 갯수를 결정할 때 선택 비율을 결정하는 변수
    size = width * height   # 영상의 가로 사이즈와 세로 사이즈를 곱한 값인 18000을 많이 사용하기 위해 선언한 영상 총 사이즈를 나타내는 변수


    # 학습파일을 저장하는 리스트 생성
    image_files = []

    # 학습파일을 리스트에 추가하여 저장
    for i in range(image_count):
        # f-string을 이용하여 0부터 309까지의 영상을 그레이스케일 영상으로 학습파일 리스트에 저장함
        image = cv2.imread(f"./face_img/train/train{i:03d}.jpg", cv2.IMREAD_GRAYSCALE)
        # 모든 영상의 사이즈가 같지 않기 때문에 resize를 통해 영상의 사이즈를 동일하게 맞춰줌
        image = cv2.resize(image, (width, height))
        # 리스트에 영상을 추가
        image_files.append(image)

    # myPCA라는 사용자 지정 함수에 테스트 영상 리스트와, 영상의 화소 갯수, 주성분 선택 비율을 전달함.
    average, difference_array, select_index, transform_matrix, pca_array = myPCA(image_files, size, rate)

    # PCA 테스트 시작
    # 얼굴 인식 테스트에 필요한 변수 선언
    test_image_count = 93   # 테스트 영상의 갯수를 나타내는 변수
    test_files = []         # 테스트 영상을 저장하는 리스트 생성
    success = 0             # 얼굴 인식 성공 여부를 나타내는 변수
    test_count = 0          # 얼굴 인식을 실행한 횟수를 저장하는 변수
    test_image_number = 0   # 테스트 영상의 번호를 나타내는 변수

    # 테스트 이미지를 리스트에 추가하여 저장
    for i in range(test_image_count):
        image = cv2.imread(f"./face_img/test/test{i:03d}.jpg", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (width, height))
        test_files.append(np.array(image, np.float32))

    while True:
        # 얼굴 인식 결과를 확인할 테스트 이미지 파일의 영상 번호를 입력받음.
        print("이미지 번호를 입력하시오.")
        while True:
            print("->", end=' ')
            test_image_number = int(input())
            # 입력한 정수형 값이 사진의 범위인 0 ~ 93, 프로그램 종료를 나타내는 -1이면 입력을 받는 반복문을 멈추고 나감
            if test_image_number < 93 and test_image_number >= -1: break
            # 잘못 입력 받은 값에 대해서는 반복을 통해 다시 입력 받아줌.
            print("이미지 번호를 잘못 입력하였습니다. 다시 입력해주세요.")
        # 입력받은 수가 -1일 경우 테스트를 종료함.
        if test_image_number == -1: break

        cv2.destroyAllWindows()

        image = test_files[test_image_number] - average
        image = image.reshape(size, 1)
        image_value = transform_matrix.T @ image

        min_array = 0
        min_number = 0

        # 영상과의 차이가 가장 작은 영상 선택
        for i in range(image_count):
            # 학습 영상을 세로영상(벡터영상)으로 변환시켜줌
            arr = pca_array[:, i].reshape(select_index, 1)
            # 학습 영상과의 차이를 저장할 변수 선언
            sum = 0
            # 선택한 주성분의 비율만큼의 값의 차이를 sum 변수에 더해줌
            for j in range(select_index):
                # 차이를 구하기 때문에 값에 -가 들어가게 되면 정확한 비교가 불가능하기 때문에 제곱을 해줌
                sum += (image_value[j, 0] - arr[j, 0]) ** 2
            # 제곱한 값을 루트를 이용하여 원래 크기로 변경함
            sum **= 1 / 2
            # 두 영상과의 차이가 기존의 최솟값 보다 작을 경우 최솟값과 번호를 변경해줌
            if i == 0 or min_array > sum:
                min_array = sum
                min_number = i

        # 가장 유사한 영상 출력
        find_image_title = f"Find Image.{min_number:03d}"
        cv2.namedWindow(find_image_title)
        cv2.moveWindow(find_image_title, 100, 100)
        cv2.imshow(find_image_title, np.array(image_files[min_number], dtype=np.uint8))

        # 테스트 영상 출력
        test_image_title = f"Test Image.{test_image_number:03d}"
        cv2.namedWindow(test_image_title)
        cv2.moveWindow(test_image_title, 200 + width, 100)
        cv2.imshow(test_image_title, np.array(test_files[test_image_number], dtype=np.uint8))

        # if test_count == 93: break
        # test_image_number += 1

        # 영상의 정확한 인식 여부를 확인하기 위해서 키입력을 받아줌
        key = cv2.waitKey()
        # 스페이스바를 입력 시 인식 성공 여부를 나타내는 success 변수에 1을 더해줌
        if key == 32:
            print(f"{test_image_number} 인식 완료")
            success += 1
        # 나머지 다른 키가 입력이 되면 인식에 실패하였다는 메시지만을 보내줌
        else:
            print(f"{test_image_number} 인식 실패")
        # 영상의 인식 횟수를 확인하기 위해 test_count 변수에 1을 더해줌
        test_count += 1
    # 프로그램이 종료되면서 인식률을 성공횟수/반복횟수 * 100을 통해 백분률로 출력을 해줌.
    print(f"인식률 : {success / test_count * 100:.2f}%")

main()

print("프로그램 종료")