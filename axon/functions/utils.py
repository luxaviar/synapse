import axon
from axon.core import Variable, as_variable, as_array, get_array_module

# y是预测值，t是标签值
def accuracy(y, t):
    """
    [WAR] This function is not differentiable.
    """
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))

# Inverted Dropout
def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)

    if axon.Config.train: # 训练模式下随机丢弃一部分神经元
        xp = get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio # 生成一个与x形状相同的随机矩阵，元素值大于dropout_ratio的为True，小于的为False
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale # 除以scale是为了在测试模式下不用为了模拟dropout而进行缩放操作
        return y
    else: # 测试模式
        return x

def embed_id(x, W):
    return W[x]