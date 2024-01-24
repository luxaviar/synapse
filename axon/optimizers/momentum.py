from axon.core import get_array_module
from axon.optimizers.base import Optimizer

# 带动量的随机梯度下降
# 每次梯度更新都会带有前几次梯度方向的惯性，使梯度的变化更加平滑，一定程度上减小权重优化过程中的震荡问题
# v = momentum * v - lr * dL/dW
# W = W + v
class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
