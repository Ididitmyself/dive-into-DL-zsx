import torch
from torch.backends import cudnn

if __name__ == '__main__':
    # 测试 CUDA
    print("Support CUDA ?: ", torch.cuda.is_available())
    x = torch.tensor([10.0])
    x = x.cuda()
    print(x)

    y = torch.randn(2, 3)
    y = y.cuda()
    print(y)

    z = x + y
    print(z)


# 测试 CUDNN
print("Support cudnn ?: ", cudnn.is_acceptable(x))
# 输出 pytorch版本
print("pytorch version is {}".format(torch.__version__))
