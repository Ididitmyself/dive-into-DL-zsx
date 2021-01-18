import torch
x = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
y = 2 * x
z = y.view(2, 2)
print("z = " + str(z))
print("-----------------------------------------------")

# 现在 z 不是一个标量，所以在调用backward时需要传入一个和z同形的权重向量进行加权求和得到一个标量。
w = torch.tensor([[1.0, 0.1], [0.01, 0.001]], dtype=torch.float64)
z.backward(w)
# z = y*w = 2*w*x
print("dz/dx = {}".format(x.grad))
print("-----------------------------------------------")

# 中断梯度追踪
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
print("是否追踪了x的计算操作：{}".format(x.requires_grad))
print("y1 = {}\n是否追踪了y1的计算操作：{}".format(y1, y1.requires_grad))
print("y2 = {}\n是否追踪了y2的计算操作：{}".format(y2, y2.requires_grad))
print("y3 = {}\n是否追踪了y3的计算操作：{}".format(y3, y3.requires_grad))
print("-----------------------------------------------")
y3.backward()
print("dy3/dx = " + str(x.grad))
print("-----------------------------------------------")

# 如果想要修改tensor的数值，但是又不希望被autograd记录（即不会影响反向传播），那么可以对tensor.data进行操作。
x = torch.ones(1, requires_grad=True)
print("x.data = " + str(x.data))  # 还是一个tensor
print("是否追踪了x.data的计算操作：{}".format(x.data.requires_grad))  # 但是已经是独立于计算图之外
y = 2 * x
x.data *= 100  # 只改变了值，不会记录在计算图，所以不会影响梯度传播
y.backward()
print("x = " + str(x))  # 更改data的值也会影响tensor的值
print("dy/dx = " + str(x.grad))