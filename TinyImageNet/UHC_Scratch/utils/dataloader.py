from torchvision import datasets, transforms
import torch
import numpy as np


def subdataset_tiny(dataset, categories):
    all_targets = np.array(dataset.targets)
    subidx = np.array([], dtype=np.int64)
    for i in categories:
        temp = np.where(all_targets == i)[0]
        subidx = np.concatenate((subidx, temp))

    dataset.imgs = [dataset.imgs[i] for i in subidx]
    dataset.samples = [dataset.samples[i] for i in subidx]
    temp_targets = all_targets[subidx]
    for i, j in enumerate(categories):
        temp_targets = np.where(temp_targets == j, i, temp_targets)

    dataset.targets = list(np.array(temp_targets, dtype=np.int64))

    for i, tup in enumerate(dataset.imgs):
        tup = list(tup)
        tup[1] = categories.index(tup[1])
        dataset.imgs[i] = tuple(tup)

    for i, tup in enumerate(dataset.samples):
        tup = list(tup)
        tup[1] = categories.index(tup[1])
        dataset.samples[i] = tuple(tup)

    return dataset


def tinyimagenet_dataloader(args, train_subidx=None, test_subidx=None):
    train_dir = "../data/tiny-imagenet-200/train"
    val_dir = "../data/tiny-imagenet-200/val"

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    if train_subidx is not None:
        train_set = subdataset_tiny(train_set, train_subidx)

    valid_set = datasets.ImageFolder(val_dir, transform=valid_transform)
    if test_subidx is not None:
        valid_set = subdataset_tiny(valid_set, test_subidx)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return train_loader, valid_loader
