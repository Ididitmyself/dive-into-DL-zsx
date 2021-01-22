import torch
from matplotlib import pyplot as plt
import numpy as np


# 将matplotlib作的图设置成嵌入显示。
# %matplotlib inline

# 设置随机数种子，使训练结果可复现。
import random
import os
def setup_torch_seed(seed):
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # # Numpy module.
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# 设置cuDNN：cudnn中对卷积操作进行了优化，牺牲了精度来换取计算效率。如果需要保证可重复性，可以使用如下设置:
# 不过实际上这个设置对精度影响不大，仅仅是小数点后几位的差别。所以如果不是对精度要求极高，其实不太建议修改，因为会使计算效率降低。
    torch.backends.cudnn.benchmark = False  # 默认是False，换成True可以加速计算
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_torch_seed(20)

# 生成数据集
feature_inputs = 2
num_examples = 1000
w = [2, -3.4]
# w = torch.tensor([2, -3.4])
b = 4.2
# randn()返回一个服从标准正态分布的张量
X = torch.randn(num_examples, feature_inputs, dtype=torch.float32)
# Y = torch.mm(X, w.T) + b
Y_hat = w[0] * X[:, 0] + w[1] * X[:, 1] + b
# 加上随机噪声。噪声项epsilon服从均值为0、标准差为0.01的正态分布。噪声代表了数据集中无意义的干扰
Y_hat += torch.tensor(np.random.normal(0, 0.01, size=Y_hat.size()), dtype=torch.float32)
print("训练集第一个样本值为：{}\n第一个样本对应的预测值为：{}".format(X[0], Y_hat[0]))

# 通过生成第二个特征features[:, 1]和标签 labels 的散点图，可以更直观地观察两者间的线性关系。
from IPython import display
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

# # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# import sys
# sys.path.append("..")
# from d2lzh_pytorch import *

set_figsize()
plt.scatter(X[:, 1].numpy(), Y_hat.numpy(), 1)
# plt.show()

# 读取数据
# 定义data_iter函数，每次返回batch_size（批量大小）个随机样本的特征和标签。
# 本函数已保存在d2lzh包中方便以后使用!!!
def data_iter(batch_size, X, Y_hat):
    num_examples = len(X)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)]) # 最后一次可能不足一个batch
        yield X.index_select(0, j), Y_hat.index_select(0, j)

batch_size = 10

for X, Y in data_iter(batch_size, X, Y_hat):
    print("随机获取的样本X = {}\n对应的Y = {}".format(X, Y))
    break

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (feature_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
# 将参数属性.requires_grad设置为True，以便追踪参数上的所有操作，利用链式法则计算梯度。
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 实现线性回归的矢量计算表达式
# 本函数已保存在d2lzh_pytorch包中方便以后使用!!!
def linreg(X, w, b):
    z = torch.mm(X, w) + b
    return z

# 定义损失函数，采用平方损失函数。需要把真实值y变形成预测值y_hat的形状。
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def squared_loss(y_hat, y):
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    loss = (y.view(y_hat.size()) - y_hat) ** 2 / 2
    return loss
# 定义优化算法，采用小批量梯度下降（Mini-Batch Gradient Descent)。
# 本函数已保存在d2lzh_pytorch包中方便以后使用!!!
def sgd(parameters, learning_rate, batch_size):
    for param in parameters:
        param.data -= learning_rate * param.grad / batch_size

# 训练模型
learning_rate = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
parameters = {
    w: w,
    b: b
}
for epoch in range(num_epochs):
    for X, Y in data_iter(batch_size, X, Y_hat):
        l = loss(net(X, w, b), Y).sum()
        l.backward()
        sgd(parameters, learning_rate, batch_size)
        # 梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(X, w, b), Y)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))
