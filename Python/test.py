# import cv2
# import numpy as np

# image = cv2.imread('/Users/lochuynhquang/Documents/thesis2022/Python/data/comga.jpeg')

# kernel1 = np.array([[-1, -1, -1],
#                     [-1,  8, -1],
#                     [-1, -1, -1]])

# edge_detect = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)
# # cv2.imshow('Original', image)
# cv2.imshow('Identity', edge_detect)


# cv2.waitKey(0) 
# cv2.destroyAllWindows()

import timeit
import numpy as np
from numpy import random as nprd
import cupy as cp
from cupy import random as cprd

def cpu():
    x_cpu = nprd.randint(0, 255,(2000,2000))
    res = np.dot(x_cpu, x_cpu)
    print(np.shape(res))

def gpu():
    x_gpu = cprd.randint(0, 255,(2000,2000))
    res = cp.dot(x_gpu, x_gpu)
    print(cp.shape(res))

starttime = timeit.default_timer()
print("The start time is :",starttime)
cpu()
print("The time difference is :", timeit.default_timer() - starttime)

