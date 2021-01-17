import torch
# 一、算数操作

# 加法
# 1.直接相加
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print("x + y = " + str(x + y))

# 2.torch.add
print("x + y = " + str(torch.add(x, y)))

# 3. 指定输出
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print("x + y = " + str(result))

# 4.inplace
# adds x to y
y.add_(x)
print("x + y = " + str(y))

# 二、索引
# 索引出来的结果与原数据共享内存，也即修改一个，另一个会跟着修改
print("索引前的x[0, :] = " + str(x[0, :]))
y = x[0, :]
y += 1
print("y = " + str(y))
print("索引后的x[0, :] = " + str(x[0, :])) # 源tensor也被改了

# 三、改变形状
# 用view()来改变Tensor的形状
y = x.view(15)
z = x.view(-1, 5)  # -1所指的维度可以根据其他维度的值推出来
print("x.size() = {} \ny.size() = {} \nz.size() = {}".format(x.size(), y.size(), z.size()))

# 注意view()返回的新Tensor与源Tensor虽然可能有不同的size，注意view()返回的新Tensor与源Tensor虽然可能有不同的size，但是是共享data的，也即更改其中的一个，另外一个也会跟着改变。
# (顾名思义，view仅仅是改变了对这个张量的观察角度，内部数据并未改变)
# 进行验证
x += 1
print("x = " + str(x))
print("y = " + str(y))  # 也加了1

# 如果我们想返回一个真正新的副本，应先用clone创造一个副本然后再使用view。
x_copy = x.clone().view(3, 5)
x -= 1
print("x = " + str(x))
print("x_copy = " + str(x_copy))
# 使用clone还有一个好处是会被记录在计算图中，即梯度回传到副本时也会传到源Tensor。

# item()函数可以将一个标量Tensor转换成一个Python number
x1 = torch.randn(1)
print("x1 = " + str(x1))
print("x1.item() = {}".format(x1.item()))