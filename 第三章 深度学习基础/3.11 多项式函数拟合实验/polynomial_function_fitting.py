import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import torch.utils.data as Data

# 生成数据集
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 打印生成的数据集的前两个样本
print("features数据集的前两个样本：{}\npoly_features数据集的前两个样本：{}\nlabels数据集的前两个样本：{}".format(features[:2], poly_features[:2], labels[:2]))
print('-------------------------------------------------------------------------------')

# 定义作图函数semilogy，其中y轴使用了对数尺度
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()
# 使用平方损失函数，使用不同复杂度的模型来拟合生成的数据集，所以我们把模型定义部分放在fit_and_plot函数中。
num_epochs, loss = 100, torch.nn.MSELoss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    model = torch.nn.Linear(train_features.shape[-1], 1)
    batch_size = min(10, train_labels.shape[0])  # 此处
    dataset = Data.TensorDataset(train_features, train_labels)
    train_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(model(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        # train_labels = train_labels.view(-1, 1)
        # test_labels = test_labels.view(-1, 1)
        # train_ls.append(loss(model(train_features), train_labels).item())
        # test_ls.append(loss(model(test_features), test_labels).item())

        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(model(train_features), train_labels).item())
        test_ls.append(loss(model(test_features), test_labels).item())
    print('最后一个epoch的训练集损失值为：', train_ls[-1], '\n最后一个epoch的测试集损失值为：', test_ls[-1])
    print('-------------------------------------------------------------------------------')
    # 绘图
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', model.weight.data, '\nbias:', model.bias.data)
    print('-------------------------------------------------------------------------------')

# 三阶多项式函数拟合（正常）
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
            labels[:n_train], labels[n_train:])

# 线性函数拟合（欠拟合）
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:])

# 训练样本不足
fit_and_plot(poly_features[:2, :], poly_features[n_train:, :],
            labels[:2], labels[n_train:])


# 测试
# print('poly_features[:n_train, :] = ', poly_features[:n_train, :])
# print('poly_features[:n_train, :] size = ', poly_features[:n_train, :].size())
# print('poly_features[:n_train, :] = ', poly_features[::])
# print('poly_features size = ', poly_features.size())