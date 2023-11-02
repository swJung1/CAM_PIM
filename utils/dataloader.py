
import torch
import numpy as np
import os
from torchvision.datasets import *
from torchvision.transforms import *
from torch.utils.data import DataLoader

# from multiprocessing import shared_memory


# For multi-processes, there could be shm memory shortage, therefore use shared memory 
# class mp_loader(datasets.CIFAR10):
#     def __init__(self, root, train, download, transform):
#         super(mp_loader, self).__init__(root=root, train=train, download=download, transform=transform)
#         self.shm = shared_memory.SharedMemory(create=True, size=self.data.nbytes)
#         data_copy = self.data
#         self.data = np.ndarray(self.data.shape, self.data.dtype, buffer=self.shm.buf)
#         self.data = data_copy

#     def shutdown_shm(self):
#         self.shm.close()
#         self.shm.unlink()
  
c10=r"""====================== CIFAR10 set ============================="""
def load_cifar10_dataset(num_samples=None, root="/home/jsw/data"):
    image_size = 32
    transforms = {
        "train": Compose([
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        "test": Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    }
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root=root,
            train=(split == "train"),
            download=True,
            transform=transforms[split],
        )
    return dataset

c100=r"""====================== CIFAR100 set ==========================="""
def load_cifar100_dataset(num_samples=None, root="/home/jsw/data"):
    image_size = 32
    transforms = {
        "train": Compose([
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        ]),
        "test": Compose([ToTensor(), Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
    }
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root=root,
            train=(split == "train"),
            download=False,
            transform=transforms[split],
        )
    return dataset

# imagenet_path=r"""====================== ImageNet set ==========================="""
# def load_imagenet_trainset(num_samples=None, data_path = "/dataset/ImageNet/Classification"):
#     data_transforms = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     """ See https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder """
#     trainset = datasets.ImageFolder(os.path.join(data_path, 'train'), data_transforms)
    
#     # extract number of images
#     trainset.samples, trainset.targets = trainset.samples[:num_samples], trainset.targets[:num_samples]
#     return trainset

# def load_imagenet_testset(num_samples=None, data_path = "/dataset/ImageNet/Classification"):
#     data_transforms = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])

#     testset = datasets.ImageFolder(os.path.join(data_path, 'val'), data_transforms)
    
#     # extract number of images
#     testset.samples, testset.targets = testset.samples[:num_samples], testset.targets[:num_samples]
#     return testset


def load_dataset(dataset_name, num_samples=None, batch_size=512, num_workers=0):
    if dataset_name == 'cifar10':
        dataset = load_cifar10_dataset(num_samples)
    elif dataset_name == 'cifar100':
        dataset = load_cifar100_dataset(num_samples)
    # elif dataset_name == 'imagenet':
    #     trainset, testset = load_imagenet_trainset(num_samples), load_imagenet_testset(num_samples)
    
    dataloader = {}
    for split in ['train', 'test']:
        dataloader[split] = DataLoader(
            dataset[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
        )
    return dataloader
                    

