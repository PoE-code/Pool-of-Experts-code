from __future__ import print_function

import torch
import torch.nn.functional as F
from utils.dataloader import ImageNet_dataloader
import torchvision.models as models
import os


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return correct / len(test_loader.dataset)


def get_Oracle(args, chk=False):
    os.makedirs('DB_pretrained/Oracle', exist_ok=True)

    _, test_loader = ImageNet_dataloader(args)

    model = models.resnet50(pretrained=True)
    model = model.to(args.device)
    model.eval()

    if chk is True:
        best_acc = test(args, model, args.device, test_loader)
        print("\nOracle Acc for all classes=%.2f%%" % (best_acc*100))

    return model
