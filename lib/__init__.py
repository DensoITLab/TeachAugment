import os

import torchvision.datasets as D
import torchvision.transforms as T


def build_dataset(dataset_name, root, train_transform=None, val_transform=None):
    if train_transform is None:
        train_transform = T.Compose([T.ToTensor()])
    if val_transform is None:
        val_transform = T.Compose([T.ToTensor()])
    if dataset_name != 'ImageNet':
        train_data = D.__dict__[dataset_name](root, download=True, transform=train_transform)
        test_data = D.__dict__[dataset_name](root, train=False, transform=val_transform)
        n_classes = 10 if dataset_name == 'CIFAR10' else 100
    else:
        train_data = D.ImageFolder(os.path.join(root, 'train'), train_transform)
        test_data = D.ImageFolder(os.path.join(root, 'val'), val_transform)
        n_classes = 1000

    return train_data, test_data, n_classes
