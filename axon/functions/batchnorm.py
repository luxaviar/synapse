import axon
from axon.core import Function, get_array_module
from axon.functions import sum

# Batch Norm的思路是调整各层的激活值分布使其拥有适当的广度
# 为此，要向神经网络中插入对数据分布进行正规化的层，好处如下：
# 1. 可以使学习快速进行（可以增大学习率）
# 2. 不那么依赖初始值（对于初始值不用那么敏感）
# 3. 抑制过拟合（降低Dropout等的必要性）
class BatchNorm(Function):
    def __init__(self, mean, var, decay, eps):
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std = None

    def forward(self, x, gamma, beta):
        assert x.ndim == 2 or x.ndim == 4

        x_ndim = x.ndim
        if x_ndim == 4:
            N, C, H, W = x.shape
            # (N, C, H, W) -> (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        xp = get_array_module(x)

        if axon.Config.train:
            mean = x.mean(axis=0) # 计算均值
            var = x.var(axis=0) # 计算方差
            inv_std = 1 / xp.sqrt(var + self.eps) # 计算标准差的倒数
            xc = (x - mean) * inv_std # 对输入数据进行标准化

            m = x.size // gamma.size # 计算输入数据的元素个数
            s = m - 1. if m - 1. > 1. else 1. # 计算无偏估计的修正值
            adjust = m / s  # unbiased estimation
            self.avg_mean *= self.decay # 更新均值
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay # 更新方差
            self.avg_var += (1 - self.decay) * adjust * var
            self.inv_std = inv_std # 保存标准差的倒数
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std
        y = gamma * xc + beta

        if x_ndim == 4:
            # (N*H*W, C) -> (N, C, H, W)
            y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy):
        gy_ndim = gy.ndim
        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gy = gy.transpose(0, 2, 3, 1).reshape(-1, C)

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        if gy_ndim == 4:
            gx = gx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return gx, ggamma, gbeta


def batch_nrom(x, gamma, beta, mean, var, decay=0.9, eps=2e-5):
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)

