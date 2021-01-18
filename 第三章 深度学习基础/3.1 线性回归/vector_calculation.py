import torch
from time import time

a = torch.ones(1000)
b = torch.ones(1000)

# 第一种向量相加方法(笨方法)：两个向量按元素逐一做标量加法
start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
end = time()
print("两个向量按元素逐一做标量加法花费的时间为：" + str(end - start))

# 第二种向量相加方法：矢量加法
start = time()
d = a + b
end = time()
print("矢量加法花费的时间为：" + str(end - start))