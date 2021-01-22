import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..")  # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

# 获取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, transform=transforms.ToTensor(), download=True)

# 打印数据集信息
print("mnist_train的类型是：" + str(type(mnist_test)))
print('mnist_train的大小为：{}\nmnist_test的大小为：{}'.format(len(mnist_train), len(mnist_test)))
print('--------------------------------------------------------------------------')

# 通过下标来访问任意一个样本
x_train_0, y_train_0 = mnist_test[0]
print('第一张图片的大小为：{}\n第一张图片的标签为：{}'.format(x_train_0.shape, y_train_0))  # Channel x Height x Width
print('--------------------------------------------------------------------------')

# 将数值标签转成相应的文本标签。
# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 定义一个可以在一行里画出多张图像和对应标签的函数
# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    # 用矢量图显示
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

# 查看训练数据集中前10个样本的图像内容和文本标签。
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 使用多进程来加速数据读取
import torch.utils.data as Data
batch_size = 256
# 判断操作系统类型
if sys.platform.startswith('win'):
    num_workers = 0 # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# 查看读取一遍训练数据需要的时间
start = time.time()
for X,y in train_iter:
    continue
# .2f保留小数后两位
end = time.time()
print('%.2f sec' % (end - start))
print('--------------------------------------------------------------------------')
