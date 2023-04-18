import numpy as np

list1, list2 = [1, 2, 3], [4, 5.0, 6]
list3 = [5.0, 'c', 3]

# 넘파이 선언
a, b, c = np.array(list1), np.array(list2), np.array(list3)

# 리스트 출력
print("리스트 출력")
print(list1)
print(list2)
list3 = list1 + list2
print(list3)

# 리스트 연산
print("리스트 연산")
print(list1 + list2)
print(list1 * 2)
# print(list1 - list2)
# print(list1 * list2)
# print(list1 / list2)

# numpy의 모든 원소는 같은 타입으로만 저장된다.
print("넘파이 출력")
print(a)
print(b)    # 원소 중 실수가 있으면 모든 원소를 실수형태로 변환한다. (n.)
print(c)    # 원소 중 문자가 있으면 모든 원소를 문자열로 변환한다.

# 넘파이 연산
print("넘파이 연산")
print(a + b)    # a의 각 원소에 b의 각 원소를 더해줌
print(a - b)    # a의 각 원소에 b의 각 원소를 빼줌
print(a * b)    # a의 각 원소에 b의 각 원소를 곱해줌
print(a / b)    # a의 각 원소에 b의 각 원소를 나눠줌
print(a // b)   # a의 각 원소에 b의 각 원소를 나눠줌(정수 나누기)
print(a % b)    # a의 각 원소에 b의 각 원소를 나눈 나머지
print(a * 2)    # a의 각 원소에 2를 더해줌
print(b + 2)    # b의 각 원소에 2를 더해줌
print(a ** b)   # a의 각 원소에 b의 각 원소만큼 제곱

# numpy 패키지
one = np.ones(10)   # 원소가 1인 넘파이 배열 선언
zero = np.zeros(9).reshape(3,3) # 원소가 0인 3 * 3 배열 선언
idt = np.identity(3)   # 3 * 3 대각 행열 선언
eye = np.eye(3, 10)    # n * m 대각 행렬 선언

# 0 ~ 1 범위의 random 수
rand1 = np.random.rand(2)    # 1차원 넘파이 배열 생성
rand2 = np.random.rand(3, 4)  # 2차원 넘파이 배열 생성
# 정수 random 수
randint1 = np.random.randint(1, 10, size=3)     # 1차원 넘파이 배열에 난수 1~10으로 생성
randint2 = np.random.randint(10, 30, size=(2, 3))   # 2차원 넘파이 배열에 난수 10~30으로 생성
# 정규 분포를 따르는 난수
randn = np.random.randn(5)  # 평균이 0이고 표준편차가 1인 난수로 생성

# 각종 numpy 패키지 출력
print("각종 numpy 패키지 출력")
print("one\n", one)
print("zero\n", zero)
print("identity\n", idt)
print("eye\n", eye)
print("rand1\n", rand1)
print("rand2\n", rand2)
print("randint1\n", randint1)
print("randint2\n", randint2)

# 행렬 변환
print("행렬 변환")
arr = np.array(range(9))
print("arr", arr)
copy = arr.copy()
flatten = arr.flatten()
ravel = arr.ravel()
reshape = arr.reshape(-1)
# reshape(-1) / reshape(1, -1) / reshape(-1,) 모두 동일
print("arr[1] = 99")
arr[1] = 99
print("arr", arr)
print("copy", copy)
print("flatten", flatten)
print("ravel", ravel)
print("reshape", reshape)
# ravel, reshape 모두 얕은 복사(주소만 참조)
''' arr [0 1 2 3 4 5 6 7 8]
    arr[1] = 99
    arr [ 0 99  2  3  4  5  6  7  8]
    copy [0 1 2 3 4 5 6 7 8]
    flatten [0 1 2 3 4 5 6 7 8]
    ravel [ 0 99  2  3  4  5  6  7  8]
    reshape [ 0 99  2  3  4  5  6  7  8] '''

# 2차원 행렬 1차원 변환
arr2 = np.array(range(12)).reshape(4, 3)
print("arr2", arr2)
flatten2 = arr2.flatten()
ravel2 = arr2.ravel()
reshape2 = arr2.reshape(-1)
print("arr2[2][:] = 99")
arr2[2][:] = 99
print("arr2", arr2)
print("flatten2", flatten2)
print("ravel2", ravel2)
print("reshape2", reshape2)
''' arr2 [[ 0  1  2]
    [ 3  4  5]
    [ 6  7  8]
    [ 9 10 11]]
    arr2[2][:] = 99
    arr2 [[ 0  1  2]
    [ 3  4  5]
    [99 99 99]
    [ 9 10 11]]
    flatten2 [ 0  1  2  3  4  5  6  7  8  9 10 11]
    ravel2 [ 0  1  2  3  4  5 99 99 99  9 10 11]
    reshape2 [ 0  1  2  3  4  5 99 99 99  9 10 11] '''