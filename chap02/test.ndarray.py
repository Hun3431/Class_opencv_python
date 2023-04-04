import numpy as np

# 3x3 2차원 배열 선언
a = [[1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]]
a_ndarray = np.array(a, int)
print(a_ndarray)

# 3x3x3 3차원 배열 선언
b = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]],
    [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
    [[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
b_ndarray = np.array(b, int)
print(b_ndarray)
