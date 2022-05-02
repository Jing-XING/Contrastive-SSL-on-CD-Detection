import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CIFAR10_IMG(Dataset):

    def __init__(self, root, train=True, transform = None, target_transform=None):
        super(CIFAR10_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        train_list=[
            'data_batch_1',
            'data_batch_2',
            'data_batch_3',

        ]
        test_list=['test_batch']

        if self.train :
            list = train_list
        else:
            list = test_list

        self.data=[]
        self.targets=[]

        for file_name in list:
            file_path = os.path.join(root, 'cifar-10-batches-py', file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)
