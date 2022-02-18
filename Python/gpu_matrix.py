import numpy as np
from numpy import random as rd
import cupy as cp



class GPUMatrix:
    def __init__(self, n,m):
        self.rows = n
        self.cols = m        
        self.data = cp.array([[0]*m] * n)      #cupy array


    def randomize(self):
        self.data = cp.array(rd.rand(self.rows, self.cols)*2 - 1)

    def show(self):
        print(cp.asarray(self.data))


    @staticmethod
    def map(m, fn):
        res = m
        for i in range(m.data.shape[0]):
            for j in range(m.data.shape[1]):
                m.data[i][j] = fn(m.data[i][j])
        return res
    
    
    @staticmethod
    def add(a,b):
        if ((a.rows == b.rows) & (a.cols == b.cols)):
            c = GPUMatrix(a.rows, a.cols)
            c.data = cp.add(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1

    @staticmethod
    def subtract(a,b):
        if ((a.rows == b.rows) & (a.cols == b.cols)):
            c = GPUMatrix(a.rows, a.cols)
            
            c.data = cp.subtract(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1
    
    @staticmethod
    def hadamard(a,b):
        if ((a.rows == b.rows) & (a.cols == b.cols)):
            c = GPUMatrix(a.rows, a.cols)
            
            c.data = cp.multiply(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1

    @staticmethod
    def multiply(a,b):
        if (a.cols == b.rows):
            c = GPUMatrix(a.rows, b.cols)
            c.data = cp.dot(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1

    @staticmethod
    def transpose(a):
        c = GPUMatrix(a.cols, a.rows)
        c.data = cp.transpose(a.data)
        return c

    @staticmethod
    def array_2_matrix(a):
        b = [a]
        c = GPUMatrix(len(a), 1)
        c.data = cp.array(b).T
        return c

    @staticmethod
    def matrix_2_array(m):
        return cp.asnumpy(cp.reshape(m.data, (1, len(m.data))))
    

    @staticmethod
    def scale(m,s):
        k = GPUMatrix(m.rows, m.cols)
        k.data = cp.multiply(m.data, s)
        return k


# k = GPUMatrix(2, 3)
# k.randomize()
# l = GPUMatrix(2, 3)
# l.randomize()
# r = GPUMatrix.add(k, l)
# k.show() 
# l.show()
# print(type(k.data))
# print(type(l.data))
# r.show()


# k = GPUMatrix.array_2_matrix([1,2,3,4,5,6])
# print(k.data)