import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# 读取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义模型
num_inputs, num_outputs, num_hiddens = 784, 10, 256

model = nn.Sequential(
    d2l.FlattenLayer(),
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs),
    )

# 定义损失函数
loss = nn.CrossEntropyLoss()

# 设置优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# 训练模型
num_epochs = 5
d2l.train_ch3(model,
              train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
