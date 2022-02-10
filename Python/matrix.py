import numpy as np
from numpy import random as rd



class Matrix:
    def __init__(self, n,m):
        self.rows = n
        self.cols = m        
        self.data = np.array([[0]*m] * n)      #numpy array


    def randomize(self):
        self.data = rd.rand(self.rows, self.cols)*2 - 1

    @staticmethod
    def map(m, fn):
        res = Matrix(m.rows, m.cols)
        res.data = map(fn, res.data)
        return res
    
    
    @staticmethod
    def add(a,b):
        if ((a.rows == b.rows) & (a.cols == b.cols)):
            c = Matrix(a.rows, a.cols)
            
            c.data = np.add(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1

    @staticmethod
    def subtract(a,b):
        if ((a.rows == b.rows) & (a.cols == b.cols)):
            c = Matrix(a.rows, a.cols)
            
            c.data = np.subtract(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1
    
    @staticmethod
    def hadamard(a,b):
        if ((a.rows == b.rows) & (a.cols == b.cols)):
            c = Matrix(a.rows, a.cols)
            
            c.data = np.multiply(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1

    @staticmethod
    def multiply(a,b):
        if (a.cols == b.rows):
            c = Matrix(a.rows, b.cols)
            c.data = np.dot(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1

    @staticmethod
    def transpose(a):
        c = Matrix(a.cols, a.rows)
        c.data = np.transpose(a.data)
        return c

    @staticmethod
    def array_2_matrix(a):
        b = [a]
        c = Matrix(len(a), 1)
        c.data = np.array(b).T
        return c

    @staticmethod
    def matrix_2_array(m):
        return np.reshape(m.data, (1, len(m.data)))
    
# k = Matrix(2, 3)
# k.randomize()
# l = Matrix(3, 4)
# l.randomize()
# r = Matrix.multiply(k, l)

# print(k.data)
# print(l.data)
# print(r.data)

# k = Matrix.array_2_matrix([1,2,3,4,5,6])
# print(k.data)


