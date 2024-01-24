import gzip
import numpy as np
import matplotlib.pyplot as plt
from axon.io import get_file
from axon.transforms import Compose, Flatten, ToFloat, Normalize
from axon.datasets.base import Dataset

# 由60000个训练样本和10000个测试样本组成，每个样本都是一张28 * 28像素的灰度手写数字图片
class MNIST(Dataset):
    def __init__(self, train=True,
                 transform=Compose([Flatten(), ToFloat(), Normalize(0., 255.)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz', # 60,000 个样本
                       'label': 'train-labels-idx1-ubyte.gz'} # 60,000 个标签
        test_files = {'target': 't10k-images-idx3-ubyte.gz', # 10,000 个样本
                      'label': 't10k-labels-idx1-ubyte.gz'} # 10,000 个标签

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # 每个样本都是一张28 * 28像素的灰度手写数字图片, -1表示自动计算样本数量
        data = data.reshape(-1, 1, 28, 28)
        return data

    # 随机展示数据集中的一副图片
    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H:(r + 1) * H, c * W:(c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)].reshape(H, W)
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}