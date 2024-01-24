from axon.optimizers.base import Optimizer
from axon.core import get_array_module

# AdaDelta是AdaGrad的改进版, 也是通过梯度平方的指数衰减移动平均来调整学习率
# 不累积全部历史梯度，而只关注过去一段时间窗口的下降梯度。这也就是AdaDelta名称中Delta的来历。
# s = rho * s + (1 - rho) * dL/dW * dL/dW (指数加权移动平均)
# dx = sqrt(msdx + eps) / sqrt(s + eps) * dL/dW (目标函数自变量的变化量)
# msdx = rho * msdx + (1 - rho) * dx * dx (指数加权移动平均)
# W = W - dx (更新自变量)
class AdaDelta(Optimizer):
    def __init__(self, rho=0.95, eps=1e-6):
        super().__init__()
        self.rho = rho
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def update_one(self, param):
        xp = get_array_module(param.data)

        key = id(param)
        if key not in self.msg:
            self.msg[key] = xp.zeros_like(param.data)
            self.msdx[key] = xp.zeros_like(param.data)

        msg, msdx = self.msg[key], self.msdx[key]
        rho = self.rho
        eps = self.eps
        grad = param.grad.data

        msg *= rho
        msg += (1 - rho) * (grad * grad)
        dx = xp.sqrt((msdx + eps) / (msg + eps)) * grad
        msdx *= rho
        msdx += (1 - rho) * dx * dx
        param.data -= dx

