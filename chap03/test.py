import numpy as np

a = np.array([1, 2, 3])

print(a)            # [1 2 3]

print(a.shape)      # (3,)  배열의 방 개수 반환

print(a.ndim)       # 1     배열의 차원 반환

print(a.dtype)      # int64 데이터 타입 반환

print(a.size)       # 3     데이터 사이즈(방 개수 반환)

print(a.itemsize)   # 8     하나의 데이터 크기 반환


b = np.zeros((2, 3, 4))
print(b)


