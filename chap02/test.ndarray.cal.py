import numpy as np

a = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]

# 3x3 2차원 리스트 선언
a_list = a
print(a_list)       #[[1, 2, 3], [4, 5, 6], [7, 8, 9]
# print(a_list + 2)
# print(a_list - 2)
print(a_list * 2)   #[[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3], [4, 5, 6], [7, 8, 9]]
# print(a_list / 2)
# list에서의 사칙연산은 list내부의 값에 접근을 하는 것이 아니라 list자체에 접근을 하기 때문에 *를 제외한 사칙연산은 사용할 수 없다.
# * 연산의 경우에는 리스트 자체를 복사해준다.

# 3x3 2차원 배열 선언
a_ndarray = np.array(a, int)
print(a_ndarray)
print(a_ndarray + 2)
print(a_ndarray - 2)
print(a_ndarray * 5)
print(a_ndarray / 2)

# 3x3x3 3차원 배열 선언
b = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
    [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
    [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
b_ndarray = np.array(b, int)
print(b_ndarray)
print(b_ndarray)
print(b_ndarray + 2)
print(b_ndarray - 2)
print(b_ndarray * 2)
print(b_ndarray / 2)