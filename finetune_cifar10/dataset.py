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
            # ONLY use 10k for train
            # 'data_batch_2',
            # 'data_batch_3',
            # 'data_batch_4',
            # 'data_batch_5',
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
        # get 5k images for train and test respectively
        temp_target = []
        temp_data = np.uint8(np.zeros((5000,3072)))
        count_list=[0,0,0,0,0,0,0,0,0,0]
        j=0
        for i in range(len(self.targets)):
            if not count_list[self.targets[i]]>=500:
                count_list[self.targets[i]]+=1
                temp_data[j]=self.data[0][i]
                j+=1
                temp_target.append(self.targets[i])
            else:
                continue
        self.targets=temp_target
        self.data[0] = temp_data


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
