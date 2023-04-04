import numpy as np

a3 = np.array([ [[0,0,0],
                 [0,0,255]],
                [[0,255,0],
                [255,255,255]]], np.uint8)

print(a3)
print()
a3[:1, :1 ,:] += np.array([0, 0, 255], np.uint8)

print(a3)
