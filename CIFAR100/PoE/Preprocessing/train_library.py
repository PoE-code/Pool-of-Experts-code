from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
import network
from utils.dataloader import get_dataloader
import os


def distillation(y, teacher_scores, T):
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T ** 2) / y.shape[0]
    return l_kl


def train(args, Oracle, student, device, train_loader, optimizer, epoch):
    Oracle.eval()
    student_library, student_classifier = student
    student_library.train()
    student_classifier.train()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        t_logit = Oracle(data)
        s_logit = student_classifier(student_library(data))

        loss = distillation(s_logit, t_logit.detach(), 4)
        loss.backward()
        optimizer.step()

        if args.verbose and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))


def test(student, device, test_loader, cur_epoch, test_only=False):
    student_library, student_classifier = student
    student_library.eval()
    student_classifier.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = student_classifier(student_library(data))
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


def get_library(args, Oracle):
    os.makedirs('DB_Pool of Experts/Library', exist_ok=True)

    train_loader, test_loader = get_dataloader(args)

    student_library = network.wresnet.wideresnet_ex(depth=16, num_classes=args.Oracle_classes,
                                                    widen_factor=1, dropRate=0.0)
    student_classifier = network.wresnet.wideresnet_cl(depth=16, num_classes=args.Oracle_classes,
                                                       EX_widen_factor=1, widen_factor=1, dropRate=0.0)

    if args.Library_pretrained is True:
        student_library.load_state_dict(torch.load('./DB_Pool of Experts/Library/library_cifar100.pt'))
        student_classifier.load_state_dict(torch.load('./DB_pretrained/student for library/classifier_cifar100.pt'))

        student_library = student_library.to(args.device)
        student_classifier = student_classifier.to(args.device)

        student_library.eval()
        student_classifier.eval()

        student = [student_library, student_classifier]
        best_acc = test(student, args.device, test_loader, 0, True)
        print("\nStudent for Library Acc=%.2f%%" % (best_acc*100))

        return student_library

    student_library = student_library.to(args.device)
    student_classifier = student_classifier.to(args.device)

    Oracle.eval()

    optimizer_S = optim.SGD(list(student_library.parameters()) + list(student_classifier.parameters()), lr=args.lr,
                            weight_decay=args.weight_decay, momentum=0.9)

    if args.scheduler:
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, [80, 160], 0.1)

    best_acc = 0

    student = [student_library, student_classifier]

    for epoch in range(1, args.Oracle_epochs + 1):
        if args.scheduler:
            scheduler_S.step()

        train(args, Oracle=Oracle, student=student, device=args.device, train_loader=train_loader,
              optimizer=optimizer_S, epoch=epoch)
        acc = test(student, args.device, test_loader, epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save(student_library.state_dict(),
                       './DB_Pool of Experts/Library/library_cifar100.pt')
            torch.save(student_classifier.state_dict(),
                       './DB_pretrained/student for library/classifier_cifar100.pt')

    print("\nStudent for Library Acc=%.2f%%" % (best_acc*100))
    student_library.load_state_dict(torch.load('./DB_Pool of Experts/Library/library_cifar100.pt'))
    student_library.eval()

    return student_library
