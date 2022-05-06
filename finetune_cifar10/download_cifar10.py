import torch 
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def main():
    batchsz = 32

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor
    ]), download=True)



if __name__ == "__main__":
    main()
