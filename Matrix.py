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
    def add(a,b):
        if (np.shape(a.data) == np.shape(b.data)):
            c = Matrix(a.rows, a.cols)
            
            c.data = np.add(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1

    def subtract(a,b):
        if (np.shape(a.data) == np.shape(b.data)):
            c = Matrix(a.rows, a.cols)
            
            c.data = np.subtract(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1
    
    def hadamard(a,b):
        if (np.shape(a.data) == np.shape(b.data)):
            c = Matrix(a.rows, a.cols)
            
            c.data = np.multiply(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1


    def multiply(a,b):
        if (np.shape(a.data)[1] == np.shape(b.data)[0]):
            c = Matrix(a.rows, b.cols)
            c.data = np.dot(a.data,b.data)
            return c
        else:
            print("wrong dims")
            return -1

    def transpose(a):
        c = Matrix(a.cols, a.rows)
        c.data = np.transpose(a.data)
        return c

    def array_2_matrix(a):
        c = Matrix(len(a), 1);
        c.data = a.transpose();
    
k = Matrix(2, 3)
k.randomize()
l = Matrix(3, 4)
l.randomize()
r = Matrix.multiply(k, l)

print(k.data)
print(l.data)
print(r.data)