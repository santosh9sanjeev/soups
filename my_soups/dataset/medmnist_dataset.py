import medmnist
import random
random.seed(42)
class PneumoniaDataset:
    def __init__(self, download=True):
        self.download = download

    def get_data(self):
        #DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(root='./data', split='train', download=self.download)

        return dataset


class OrganCMNIST:
    def __init__(self, download=True):
        self.download = download

    def get_data(self):
        DataClass = getattr(medmnist, info['python_class'])
        #dataset = DataClass(root='./data', split='train', download=self.download)

        return DataClass