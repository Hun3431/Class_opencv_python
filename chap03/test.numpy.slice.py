import numpy as np

# 1차원 numpy 배열
a = np.array([1, 2, 3, 4, 5])

print(a[1])
print(a[:3:2])
print(a[1:4])
print(a[1:4:2])
print(a[3:0:-1])
print(a[3:0:-2])
print(a[-1:])

# 2차원 numpy 배열
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

print(b, b.shape)                  # 2차원
print(b[0], b[0].shape)            # 1차원
print(b[0:2], b[0:2].shape)        # 2차원
print(b[1, 0:2], b[1, 0:2].shape)  # 1차원
print(b[1, 0], b[1, 0].shape)      # scalar

print(b[1:2, 1:2], b[1:2, 1:2].shape)
print(b[1:3, 1:2], b[1:3, 1:2].shape)