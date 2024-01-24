import numpy as np
import axon.functions as F
from axon.core import Parameter, get_array_module, cupy
from axon.layers.base import Layer

# 全连接层
class Linear(Layer):
    # in_size可以为None, 此时需要在forward中初始化W
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self, xp=cupy):
        I, O = self.in_size, self.out_size
        # 采用Xavier初始值
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y
