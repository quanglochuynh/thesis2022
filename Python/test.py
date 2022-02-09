# import numpy as np
# import matplotlib.pyplot as plt 

# x = np.linspace(-1,1, 100);

# y = np.multiply(x,x);
# y = np.multiply(y,x);

# fig = plt.figure()
# ax = fig.subplots()
# ax.plot(x,y, color = 'b', label = 'Data points') 

# plt.show()

import numpy as np
from numpy import random as rd

x=rd.rand(5,10)
x = 2*x
x = x-1
print(x)