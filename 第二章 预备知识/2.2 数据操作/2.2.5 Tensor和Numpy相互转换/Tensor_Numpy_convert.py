import torch

# tensor转numpy
# 我们很容易用numpy()和from_numpy()将Tensor和NumPy中的数组相互转换。
# 但是需要注意的一点是： 这两个函数所产生的的Tensor和NumPy中的数组共享相同的内存（所以他们之间的转换很快），改变其中一个时另一个也会改变！！！

# 1.使用numpy()将Tensor转换成NumPy数组
a = torch.ones(5)
b = a.numpy()
print("a = {}\nb = {}".format(a, b))
print("-----------------------------------------")
a += 1
print("a = {}\nb = {}".format(a, b))
print("-----------------------------------------")
b += 1
print("a = {}\nb = {}".format(a, b))
print("-----------------------------------------")

# 2.使用from_numpy()将NumPy数组转换成Tensor
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
print("a = {}\nb = {}".format(a, b))
print("-----------------------------------------")

a += 1
print("a = {}\nb = {}".format(a, b))
print("-----------------------------------------")

b += 1
print("a = {}\nb = {}".format(a, b))
print("-----------------------------------------")

# 3.直接用torch.tensor()将NumPy数组转换成Tensor
# 需要注意的是该方法总是会进行数据拷贝，返回的Tensor和原来的数据不再共享内存。
c = torch.tensor(a)
a += 1
print("a = {}\nc = {}".format(a, c))
print("-----------------------------------------")
