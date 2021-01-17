# 导入PyTorch
import torch

# 一、创建Tensor

# 1.创建一个5x3的未初始化的Tensor
x1 = torch.empty(5, 3)
print("x1 = " + str(x1))

# 2.创建一个5x3的随机初始化的Tensor
x2 = torch.rand(5, 3)
print("x2 = " + str(x2))

# 3.创建一个5x3的long型全0的Tensor，long为长整型数据，含4个字节，取值范围为：-2^31 ~ (2^31 -1)。
x3 = torch.zeros(5, 3, dtype=torch.long)
print("x3 = " + str(x3))

# 4.直接根据数据创建

# x4 = torch.tensor(5.5, 3)
# TypeError: tensor() takes 1 positional argument but 2 were given
# tensor中填多个数字要用list表示
x4 = torch.tensor([5.5, 3])
print("x4 = " + str(x4))

# 5.通过现有的Tensor来创建，此方法会默认重用输入Tensor的一些属性，例如数据类型，除非自定义数据类型。
x5 = x3.new_ones(5, 3, dtype=torch.float64)  # 返回的tensor默认具有相同的torch.dtype和torch.device
print("x5 = " + str(x5))

x6 = torch.randn_like(x4, dtype=torch.float) # 指定新的数据类型
print("x6 = " + str(x6))

# 二、通过shape或者size()来获取Tensor的形状

# print("x5的大小为：" + x5.size())
# TypeError: can only concatenate str (not "torch.Size") to str
# 改正：
print("x5的大小为：" + str(x5.size()))
print("x6的大小为：" + str(x6.shape))
# 注意：返回的torch.Size其实就是一个tuple, 支持所有tuple的操作
