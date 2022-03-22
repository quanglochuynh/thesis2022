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