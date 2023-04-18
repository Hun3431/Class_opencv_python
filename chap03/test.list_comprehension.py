import numpy as np

abc = [[i-j for i in range(10,-4,-2) if abs(i) % 3 != 1] for j in range(10) if j % 2 == 0]
print(abc)
# [[8, 6, 2, 0, -2], [6, 4, 0, -2, -4], [4, 2, -2, -4, -6], [2, 0, -4, -6, -8], [0, -2, -6, -8, -10]]
print(np.array(abc))
''' [[  8   6   2   0  -2]
     [  6   4   0  -2  -4]
     [  4   2  -2  -4  -6]
     [  2   0  -4  -6  -8]
     [  0  -2  -6  -8 -10]] '''


arr = [[i-j for i in range(j,-4,-2) if abs(i) % 3 != 1][:4] for j in range(10) if j % 2 == 0 if j > 5]
print(arr)
# [[0, -4, -6, -8], [0, -2, -6, -8]]
print(np.array(arr))
''' [[ 0 -4 -6 -8]
     [ 0 -2 -6 -8]] '''

