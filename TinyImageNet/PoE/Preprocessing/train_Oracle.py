from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
import network
from utils.dataloader import tinyimagenet_dataloader
import os


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if args.verbose and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))


def test(args, model, device, test_loader, cur_epoch, test_only=False):
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

    if test_only:
        return correct / len(test_loader.dataset)

    print('\nEpoch {} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        cur_epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)


def get_Oracle(args):
    os.makedirs('DB_pretrained/Oracle', exist_ok=True)

    train_loader, test_loader = tinyimagenet_dataloader(args)
    model = network.wresnet.wideresnet(depth=16, num_classes=args.Oracle_classes, widen_factor=10, dropRate=0.0)

    if args.Oracle_pretrained is True:
        model.load_state_dict(torch.load('./DB_pretrained/Oracle/Oracle_TinyImageNet.pt'))
        model = model.to(args.device)
        model.eval()
        best_acc = test(args, model, args.device, test_loader, 0, True)
        print("\nOracle Acc for all classes=%.2f%%" % (best_acc*100))
        return model

    model = model.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    best_acc = 0

    if args.scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.Oracle_step_size, 0.1)

    for epoch in range(1, args.Oracle_epochs + 1):
        if args.scheduler:
            scheduler.step()

        train(args, model, args.device, train_loader, optimizer, epoch)
        acc = test(args, model, args.device, test_loader, epoch)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.ckpt)

    print("\nOracle Acc for all classes = %.2f%%" % (best_acc*100))
    model.load_state_dict(torch.load('./DB_pretrained/Oracle/Oracle_cifar100.pt'))
    model.eval()
    return model
