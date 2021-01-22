# 高维线性回归问题，高维是指输入特征有很多很多
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import torch.utils.data as Data

# 生成数据集
n_train, n_test, num_inputs = 20, 100, 200

# W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float)
# b = torch.zeros(num_outputs, dtype=torch.float)
true_b = 0.05
true_w = torch.ones((num_inputs, 1), dtype=torch.float) * 0.01

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 实现权重衰减的方法，通过在目标函数后添加L2范数惩罚项来实现权重衰减

# 初始化模型参数
def init_params():
    W = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # print(b)
    return [W, b]

# 定义L2范数惩罚项
def l2_penalty(W):
    return (W**2).sum() / 2

# 定义数据集
batch_size, num_epochs, lr = 1, 100, 0.003
model, loss = d2l.linreg, d2l.squared_loss

dataset = Data.TensorDataset(train_features, train_labels)
train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot(lambd):
    parameters = init_params()
    W = parameters[0]
    b = parameters[1]
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(model(X, W, b), y) + lambd * l2_penalty(W)
            l = l.sum()

            if W.grad is not None:
                W.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd(parameters, lr, batch_size)
        train_ls.append(loss(model(train_features, W, b), train_labels).mean().item())
        test_ls.append(loss(model(test_features, W, b), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs+1), test_ls, legend=['train', 'test'])
    print('L2 norm of W:', W.norm().item())  # 返回W的2-范数

# 训练并测试高维线性回归模型
# 当lambd设为0时，我们没有使用权重衰减。结果训练误差远小于测试集上的误差。这是典型的过拟合现象。
fit_and_plot(lambd=0)
# 使用权重衰减
# 可以看出，训练误差虽然有所提高，但测试集上的误差有所下降。过拟合现象得到一定程度的缓解。
# 另外，权重参数的L2范数比不使用权重衰减时的更小，此时的权重参数更接近0。
fit_and_plot(lambd=3)