from axon.optimizers.base import Optimizer

# 随机梯度下降
# W = W - lr * dL/dW
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data

