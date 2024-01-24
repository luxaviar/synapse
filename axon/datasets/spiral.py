import numpy as np
from axon.datasets.base import Dataset

def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    # 每类样本数, 种类, 维度
    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int32)

    for j in range(num_class): # 类别
        for i in range(num_data): # 每类样本数
            # 0~1之间
            rate = i / num_data
            # 半径
            radius = 1.0 * rate
            # 角度
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            # 坐标
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t

class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)

