from torchvision import datasets, transforms
import torch
import numpy as np


def subdataset(dataset, categories):
    all_targets = np.array(dataset.targets)
    subidx = np.array([], dtype=np.int64)
    for i in categories:
        temp = np.where(all_targets == i)[0]
        subidx = np.concatenate((subidx, temp))

    dataset.data = dataset.data[subidx]
    temp_targets = all_targets[subidx]
    for i, j in enumerate(categories):
        temp_targets = np.where(temp_targets == j, i, temp_targets)

    dataset.targets = list(np.array(temp_targets, dtype=np.int64))

    return dataset


def get_dataloader(args, train_subidx=None, test_subidx=None, download=False):
    dataset_train = datasets.CIFAR100(args.data_root, train=True, download=download,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                      ]))

    if train_subidx is not None:
        dataset_train = subdataset(dataset_train, train_subidx)

    dataset_test = datasets.CIFAR100(args.data_root, train=False, download=download,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                     ]))

    if test_subidx is not None:
        dataset_test = subdataset(dataset_test, test_subidx)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)

    return train_loader, test_loader