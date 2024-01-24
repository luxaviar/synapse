import tarfile
import pickle
import numpy as np
import matplotlib.pyplot as plt
from axon.io import get_file
from axon.transforms import Compose, ToFloat, Normalize
from axon.datasets.base import Dataset
from axon.io import save_cache_npz, load_cache_npz

# 该数据集共有60000张彩色图像，这些图像是32*32，分为10个类，每类6000张图
# 50000张用于训练，构成了5个训练批，每一批10000张图
# 10000用于测试，单独构成一批
class CIFAR10(Dataset):
    def __init__(self, train=True,
                 transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 target_transform=None):
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.data, self.label = load_cache_npz(url, self.train)
        if self.data is not None:
            return
        filepath = get_file(url)
        if self.train:
            self.data = np.empty((50000, 3 * 32 * 32))
            self.label = np.empty((50000), dtype=np.int)
            for i in range(5):
                self.data[i * 10000:(i + 1) * 10000] = self._load_data(
                    filepath, i + 1, 'train')
                self.label[i * 10000:(i + 1) * 10000] = self._load_label(
                    filepath, i + 1, 'train')
        else:
            self.data = self._load_data(filepath, 5, 'test')
            self.label = self._load_label(filepath, 5, 'test')
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz(self.data, self.label, url, self.train)


    def _load_data(self, filename, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    data = data_dict[b'data']
                    return data

    def _load_label(self, filename, idx, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if ('data_batch_{}'.format(idx) in item.name and data_type == 'train') or ('test_batch' in item.name and data_type == 'test'):
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    return np.array(data_dict[b'labels'])

    def show(self, row=10, col=10):
        H, W = 32, 32
        img = np.zeros((H*row, W*col, 3))
        for r in range(row):
            for c in range(col):
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[np.random.randint(0, len(self.data)-1)].reshape(3,H,W).transpose(1,2,0)/255
        plt.imshow(img, interpolation='nearest')
        plt.axis('off')
        plt.show()

    @staticmethod
    def labels():
        return {0: 'ariplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

# 类似于CIFAR-10，不同之处在于它有100个类别，每个类别包含600张图像
# 每个类别的图像划分为500张训练图像和100张测试图像
# 100个类别分为20个超类super-class。每个图像都带有一个“精细”标签（它所属的类）和一个“粗糙”标签（它所属的超类）。
class CIFAR100(CIFAR10):
    def __init__(self, train=True,
                 transform=Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
                 target_transform=None,
                 label_type='fine'):
        assert label_type in ['fine', 'coarse']
        self.label_type = label_type
        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        self.data, self.label = load_cache_npz(url, self.train)
        if self.data is not None:
            return

        filepath = get_file(url)
        if self.train:
            self.data = self._load_data(filepath, 'train')
            self.label = self._load_label(filepath, 'train')
        else:
            self.data = self._load_data(filepath, 'test')
            self.label = self._load_label(filepath, 'test')
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz(self.data, self.label, url, self.train)

    def _load_data(self, filename, data_type='train'):
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    data = data_dict[b'data']
                    return data

    def _load_label(self, filename, data_type='train'):
        assert data_type in ['train', 'test']
        with tarfile.open(filename, 'r:gz') as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding='bytes')
                    if self.label_type == 'fine':
                        return np.array(data_dict[b'fine_labels'])
                    elif self.label_type == 'coarse':
                        return np.array(data_dict[b'coarse_labels'])

    @staticmethod
    def labels(label_type='fine'):
        coarse_labels = dict(enumerate(['aquatic mammals','fish','flowers','food containers','fruit and vegetables','household electrical device','household furniture','insects','large carnivores','large man-made outdoor things','large natural outdoor scenes','large omnivores and herbivores','medium-sized mammals','non-insect invertebrates','people','reptiles','small mammals','trees','vehicles 1','vehicles 2']))
        fine_labels = []
        return fine_labels if label_type == 'fine' else coarse_labels