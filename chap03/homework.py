import math
import numpy as np

def vector_len(a, b):
    a = math.fabs(a-9)
    b = math.fabs(b-9)

    c = math.sqrt(math.pow(a, 2) + math.pow(b, 2))
    return c

a = np.full((20,20), 0)

# for i in range(0, 20):
#     for j in range(0, 20):
#         c = vector_len(i, j)
#         if c>3 and c<5 :
#             a[i][j] = 1

a = np.array([np.array([(1 if ((vector_len(i,j)>3) and (vector_len(i, j)<5)) else 0) for i in range(0, 20) ])for j in range(0, 20)])

print(a)
