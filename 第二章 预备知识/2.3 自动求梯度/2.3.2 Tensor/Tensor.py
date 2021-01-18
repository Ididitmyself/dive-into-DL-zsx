import torch
# 创建一个Tensor并设置requires_grad=True
x = torch.ones(2, 2, requires_grad=True)
print("x = " + str(x))
print("x.grad_fn = {}".format(x.grad_fn))
print("-----------------------------------------------")
# Function是另外一个很重要的类。Tensor和Function互相结合就可以构建一个记录有整个计算过程的有向无环图（DAG）。
# 每个Tensor都有一个.grad_fn属性，该属性即创建该Tensor的Function, 就是说该Tensor是不是通过某些运算得到的?
# 若是，则grad_fn返回一个与这些运算相关的对象，否则是None。
y = x + 2
print("y = " + str(y))
print("y.grad_fn = {}".format(y.grad_fn))
print("-----------------------------------------------")
# 注意x是直接创建的，所以它没有grad_fn, 而y是通过一个加法操作创建的，所以它有一个为<AddBackward>的grad_fn.
# 像x这种直接创建的称为叶子节点，叶子节点对应的grad_fn是None。
print("x是否为叶子节点：{}\ny是否为叶子节点：{}".format(x.is_leaf, y.is_leaf))
print("-----------------------------------------------")

z = y * y * 3
out = z.mean()
print(z, out)
print("-----------------------------------------------")

# 通过.requires_grad_()来用in-place的方式改变requires_grad属性
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print("a = " + str(a))
print("a.requires_grad = {}".format(a.requires_grad))
a.requires_grad_(True)
print("a.requires_grad = {}".format(a.requires_grad))
b = (a * a).sum()
print("b = " + str(b))
print("b.grad_fn = {}".format(b.grad_fn))
print("-----------------------------------------------")

print(x.grad)