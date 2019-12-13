# import sys
# sys.path.append('/path/new/pythonfile')
import numpy as np




gen = (x**2 for x in range (1,5))
cc = sum(x**2 for x in range (1,5))

cc1 = np.arange(15)

a = np.array([[1,2,3],[5,6,7]])
b = np.array([[1,4,3],[3,6,8]])
dd= b.copy()
dd[a==b]=-1
# print(~(a == b))
# print(dd)

a1 = np.empty((5,4))
a1[0]=1
a1.astype(dtype=float)


de = np.empty((8,5))
for i in range(8):
    de[i] = i
h1 = de[[0,3,6]]
h2 = np.random.randn(4,4).reshape(8,2)
h2.transpose()

h3=np.random.rand(4,3)
h4=np.random.rand(4,3)
h5=np.maximum(h3,h4)
print(h5)