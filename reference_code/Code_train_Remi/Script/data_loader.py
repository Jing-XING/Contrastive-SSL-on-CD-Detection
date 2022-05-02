# from utils import plot_images
from sklearn.model_selection import KFold, ShuffleSplit
from torchvision.datasets.folder import *
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import random
from torch.utils.data.dataset import Dataset
import cv2
import pandas as pd
from utils import *


def get_train_valid_loader_consensus(csv, batch_size=16,
                                     random_seed=88,
                                     subset='test',
                                     shuffle=True,
                                     show_sample=False,
                                     num_workers=4,
                                     pin_memory=False,
                                     mode='KFold', n_splits=10, strong_test=False, test_size=0.2,
                                     valid_size=0.1, phase=4, expert_level='all', remove_NC=False):
    """
        Utility function for loading and returning train and valid
        multi-process iterators over the MNIST dataset. A sample
        9x9 grid of the images can be optionally displayed.

        If using CUDA, num_workers should be set to 1 and pin_memory to True.

        Args
        ----
        - data_dir: path directory to the dataset.
        - batch_size: how many samples per batch to load.
        - random_seed: fix seed for reproducibility.
        - valid_size: percentage split of the training set used for
          the validation set. Should be a float in the range [0, 1].
          In the paper, this number is set to 0.1.
        - shuffle: whether to shuffle the train/validation indices.
        - show_sample: plot 9x9 sample grid of the dataset.
        - num_workers: number of subprocesses to use when loading the dataset.
        - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
          True if using GPU.

        Returns
        -------
        - train_loader: training set iterator.
        - valid_loader: validation set iterator.
        """

    # define transforms
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True

    if subset == 'train':
        dataset = dataset_consensus(csv, do_transform=True)
    elif subset == 'valid':
        dataset = dataset_consensus(csv, do_transform=False)
    elif subset == 'test':
        dataset = dataset_consensus(csv, do_transform=False)
    else:
        raise ValueError("subset have to be train, valid or test")

    trainloader_list = []
    testloader_list = []
    validloader_list = []
    if mode == 'KFold':
        if test_size != 0:
            kf = KFold(n_splits=int(1 / test_size), shuffle=True, random_state=random_seed)
        else:
            kf = KFold(n_splits=1, shuffle=True, random_state=random_seed)
    elif mode == 'ShuffleSplit':
        kf = ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=None, random_state=random_seed)
    else:
        raise ValueError(f"Mode have to be KFold or ShuffleSplit (here {mode})")

    for i, (train_index, test_index) in enumerate(kf.split(dataset)):

        num_train = len(train_index)
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.shuffle(train_index)
            np.random.shuffle(test_index)

        train_indx, valid_index = train_index[split:], train_index[:split]

        train = torch.utils.data.Subset(dataset, train_indx)
        test = torch.utils.data.Subset(dataset, test_index)
        valid = torch.utils.data.Subset(dataset, valid_index)

        trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                  pin_memory=False)
        testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                 pin_memory=False)
        validloader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                  pin_memory=False)
        # for (x, y, path) in trainloader:
        #     print('x : {} | y : {} | path : {}'.format(np.shape(x), np.shape(y), (path)))

        trainloader_list.append(trainloader)
        testloader_list.append(testloader)
        validloader_list.append(validloader)

    if subset == 'train':
        return trainloader_list
    elif subset == 'valid':
        return validloader_list
    elif subset == 'test':
        return testloader_list
    else:
        raise ValueError("subset have to be train, valid or test")

class dataset_consensus(Dataset):
    def __init__(self, csv_file, do_transform=True, random_seed=88):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.do_transform = do_transform
        self.random_seed = random_seed

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        trans_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation((180, 180)),
            transforms.ToTensor()
        ])
        trans_resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        no_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.landmarks_frame.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.landmarks_frame.iloc[idx, 1]
        # image = np.transpose(image, )

        # sample = {'image': image, 'saliency map': saliency_map, 'label': label, 'im path': img_name}
        if self.do_transform:
            random.seed(self.random_seed)  # apply this seed to img tranfsorm
            image_trans = trans_train(image)
            image = trans_resize(image_trans)
            sample = [image, label, img_name]
        else:
            image_trans = no_trans(image)
            image = trans_resize(image_trans)
            sample = [image, label, img_name]

        return sample



