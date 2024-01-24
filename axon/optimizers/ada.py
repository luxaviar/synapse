from axon.optimizers.base import Optimizer
from axon.core import get_array_module

# Adpative Gradient
# h = h + dL/dW * dL/dW (保存了以前的所有梯度值的平方和)
# W = W - lr * dL/dW / (sqrt(h) + eps) (通过sqrt(h)调整学习的尺度)
# 这意味着，参数的元素中变动较大（被大幅更新）的元素的学习率将变小。
# 也就是说，可以按参数的元素进行学习率衰减，使变动大的参数的学习率逐渐减小。
class AdaGrad(Optimizer):
    def __init__(self, lr=0.001, eps=1e-8):
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        xp = get_array_module(param.data)

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        h = self.hs[h_key]

        h += grad * grad
        param.data -= lr * grad / (xp.sqrt(h) + eps)