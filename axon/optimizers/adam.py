import math
from axon.optimizers.base import Optimizer
from axon.core import get_array_module

# Adam直觉上看，就像是 Momentum 和 AdaGrad 的融合, 也是通过梯度平方的指数衰减移动平均来调整学习率， 但它还使用了动量项
# m = beta1 * m + (1 - beta1) * dL/dW (指数加权移动平均)
# v = beta2 * v + (1 - beta2) * dL/dW * dL/dW (指数加权移动平均)
# lr = alpha * sqrt(1 - beta2^t) / (1 - beta1^t) (计算学习率)
# W = W - lr * m / (sqrt(v) + eps) (更新自变量)
class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param):
        xp = get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        param.data -= self.lr * m / (xp.sqrt(v) + eps)