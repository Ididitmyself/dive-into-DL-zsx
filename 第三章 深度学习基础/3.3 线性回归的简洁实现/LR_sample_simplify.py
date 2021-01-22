from d2lzh_pytorch import *

# 设置随机数种子
setup_torch_seed(1)

# 数据预处理
# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs, dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 读取数据，PyTorch提供了data包来读取数据
import torch.utils.data as Data
# 设置批量大小为10
batch_size = 10
# 构造数据集：将训练数据的样本和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

# 读取并打印第一个小批量数据样本
for X, y in data_iter:
    print("第一个小批量数据为：{}\n对应标签为：{}".format(X, y))
    print("-----------------------------------------------------")
    break

# 定义模型
# 用nn.Module实现一个线性回归模型，通过继承nn.Module，撰写自己的网络/层
class LinearNet(nn.Module):
    def __init__(self, n_feature):  # 初始化
        super(LinearNet, self).__init__()  # 继承父类中的init方法
        self.linear = nn.Linear(n_feature, 1)   # 属性
    # forward 定义前向传播
    def forward(self, x):   # 方法
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)  # 实例化时，直接传入参数
print("网络 = " + str(net))  # 使用print可以打印出网络的结构
print("-----------------------------------------------------")

# 通过net.parameters()来查看模型所有的可学习参数，此函数将返回一个generator。
for param in net.parameters():
    print("可学习参数有：" + str(param))
print("-----------------------------------------------------")

# print(dir(net.parameters())) 用dir函数可以知道一个模块里面提供了哪些可以调用的函数和类
# help(net.parameters()) 用help函数了解某个函数或者类的具体用法

# 初始化模型参数
from torch.nn import init

init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

# 使用nn.Module提供的均方误差损失作为模型的损失函数。
loss = nn.MSELoss()

# 定义优化算法
import torch.optim as optim

# 创建一个用于优化net所有参数的优化器实例，并指定学习率为0.03的小批量随机梯度下降（SGD）为优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print("优化器 = " + str(optimizer))
print("-----------------------------------------------------")

# # 还可以为不同子网络设置不同的学习率，这在finetune时经常用到。
# optimizer = optim.SGD([
#     # 如果对某个参数不指定学习率，就使用最外层的默认学习率
#     {'params': net.subnet1.parameters()},  # lr=0.03
#     {'params': net.subnet2.parameters(), 'lr': 0.01}
# ], lr=0.03)

# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1  # 学习率变为之前的0.1倍

# 训练模型
num_epochs = 10
# for in 循环：经常用于遍历字符串、列表，元组，字典等可迭代对象，就是可遍历对象
# 在此，num_epochs和数字不是可迭代对象
for epoch in range(0, num_epochs):
    for X, y in data_iter:
        batch_input = net(X)
        l = loss(batch_input, y.view(-1, 1))  # -1在这里的意思是让电脑帮我们计算
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('第 %d 次epoch, loss值为 %f' % (epoch, l.item()))
print("-----------------------------------------------------")

# 比较学到的模型参数和真实的模型参数
dense = net.linear
print("真实的模型参数w为：{}\n学到的模型参数w为：{}".format(true_w, dense.weight))
print("真实的模型参数b为：{}\n学到的模型参数b为：{}".format(true_b, dense.bias))
print("-----------------------------------------------------")
