from d2lzh_pytorch import *
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
sys.path.append("..")


# 设置随机数种子
setup_torch_seed(20)

# 数据预处理
# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float)

print("第一个样本的特征向量为：{}\n第一个样本对应的标签true值为：{}".format(features[0], labels[0]))
print("-----------------------------------------------------")
# def use_svg_display():
#     # 用矢量图显示
#     display.set_matplotlib_formats('svg')

# def set_figsize(figsize=(3.5, 2.5)):
#     use_svg_display()
#     # 设置图的尺寸
#     plt.rcParams['figure.figsize'] = figsize

# 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# import sys
# sys.path.append("..")
# from d2lzh_pytorch import *

# 通过生成第二个特征features[:, 1]和标签 labels 的散点图，可以更直观地观察两者间的线性关系。
set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
plt.show()

# 本函数已保存在d2lzh包中方便以后使用
# def data_iter(batch_size, features, labels):
#     num_examples = len(features)
#     indices = list(range(num_examples))
#     random.shuffle(indices)  # 样本的读取顺序是随机的
#     for i in range(0, num_examples, batch_size):
#         j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
#         yield  features.index_select(0, j), labels.index_select(0, j)

# 设置批量大小为10
batch_size = 10

# 读取第一个小批量数据样本并打印
for X, y in data_iter(batch_size, features, labels):
    print("第一个小批量数据为：{}\n对应标签为：{}".format(X, y))
    print("-----------------------------------------------------")
    break

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# 保持梯度追踪
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 定义模型，本函数已保存在d2lzh_pytorch包中方便以后使用
# def linreg(X, w, b):
#     return torch.mm(X, w) + b

# 定义损失函数，本函数已保存在d2lzh_pytorch包中方便以后使用
# def squared_loss(y_hat, y):
#     # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
#     return (y_hat - y.view(y_hat.size())) ** 2 / 2

# 定义优化算法，本函数已保存在d2lzh_pytorch包中方便以后使用
# def sgd(params, lr, batch_size):
#     for param in params:
#         param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data

# 训练模型

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('第 %d 次epoch, loss值为 %f' % (epoch + 1, train_l.mean().item()))
print("-----------------------------------------------------")

# 比较学到的参数和用来生成训练集的真实参数
print('最优参数w值为：{}\n学习后的参数w值为：{}'.format(true_w, w))
print('最优参数b值为：{}\n学习后的参数b值为：{}'.format(true_b, b))
print("-----------------------------------------------------")