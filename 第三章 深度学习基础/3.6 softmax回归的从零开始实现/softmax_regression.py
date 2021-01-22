import torch
import torchvision
import numpy as np
import sys
# sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

# matplotlib 画图使用中文字体
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
mpl.rcParams['font.size'] = 12  # 字体大小
mpl.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 初始化模型参数
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)  # 没有dtype=torch.float就会导致数据类型不一致，报错：RuntimeError: expected scalar type Float but found Double
b = torch.zeros(num_outputs, dtype=torch.float)

# 记录参数梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

# 测试softmax函数
# X = torch.rand((2, 5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(dim=1))

# 定义模型
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

# 定义损失函数

# 测试gather函数：得到标签的预测概率
# n = 0
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = torch.LongTensor([1, 2, 3])
# print(y.shape[0])
# n += y.shape[0]
# print(n)
# print(y_hat)
# print(y.view(2, 1))
# print(y_hat.gather(1, y.view(2, 1)))  # 等同于y_hat.gather(1, y.view(-1, 1))

# 交叉熵损失函数
def cross_entropy(y_hat, y):
    # 使用gather函数得到标签的预测概率
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

# 计算分类准确率
# y_hat.argmax(dim=1)返回矩阵y_hat每行中最大元素的索引，且返回结果与变量y形状相同。
def accuracy(y_hat, y):
    # a = y_hat.argmax(dim=1) == y
    # print(a)
    # b = a.float()
    # print(b)
    # c = b.mean()
    # print(c)
    # d = c.item() # 一个元素张量可以用item得到元素值
    # print(d)
    # return d
    return (y_hat.argmax(dim=1) == y).float().mean().item()

# accuracy(y_hat, y)

# 类似地，我们可以评价模型net在数据集data_iter上的准确率。
# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

# 训练模型
num_epochs, lr = 6, 0.1
# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()  # loss为cross_entropy

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
print('-------------------------------------------------------------------------------')

# 预测
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
prediction_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = ['预测:' + prediction + '\n真实:' + true for prediction, true in zip(prediction_labels, true_labels)]

d2l.show_fashion_mnist(X[0:10], titles[0:10])




