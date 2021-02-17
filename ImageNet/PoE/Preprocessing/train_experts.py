from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
import network
from utils.dataloader import ImageNet_dataloader
from utils.ImageNet_hierarchy import ImageNet_Superclass
import os


def distillation(y, teacher_scores, T):
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T ** 2) / y.shape[0]
    return l_kl


def train(args, Oracle, student, device, train_loader, optimizer, epoch, priTask_idx):
    Oracle.eval()
    library, expert = student
    library.train()
    expert.train()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        t_logit = Oracle(data)
        s_logit = expert(library(data).detach())

        soft_loss = distillation(s_logit, t_logit[:, priTask_idx].detach(), 4)
        scale_loss = F.l1_loss(s_logit, t_logit[:, priTask_idx].detach())
        loss = soft_loss + (args.alpha * scale_loss)
        loss.backward()
        optimizer.step()

        if args.verbose and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))


def test(student, device, test_loader, cur_epoch, test_only=False):
    library, expert = student
    library.eval()
    expert.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = expert(library(data))
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


def get_experts(args, Oracle, library):
    os.makedirs('DB_Pool of Experts/Experts', exist_ok=True)

    for primitiveTask in ImageNet_Superclass.keys():
        priTask_idx = ImageNet_Superclass[primitiveTask]

        train_loader, test_loader = ImageNet_dataloader(args, test_subidx=priTask_idx)

        expert = network.pytorch_mobilenetV2.mobilenet_cl(outchannel=40, num_classes=len(priTask_idx))

        priTask_path = './DB_Pool of Experts/Experts/expert_%s.pt' % primitiveTask

        if args.Experts_pretrained is True:
            if not os.path.exists(priTask_path):
                continue
            expert.load_state_dict(torch.load(priTask_path))
            expert = expert.to(args.device)

            library.eval()
            expert.eval()

            student = [library, expert]
            best_acc = test(student, args.device, test_loader, 0, True)
            print("\nModel for %s Acc=%.2f%%" % (primitiveTask, best_acc*100))
            continue

        expert = expert.to(args.device)

        Oracle.eval()
        library.eval()

        optimizer_S = optim.SGD(expert.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)

        if args.scheduler:
            scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, [10, 20], 0.1)

        best_acc = 0

        student = [library, expert]

        for epoch in range(1, args.Experts_epochs + 1):
            if args.scheduler:
                scheduler_S.step()

            train(args, Oracle=Oracle, student=student, device=args.device, train_loader=train_loader,
                  optimizer=optimizer_S, epoch=epoch, priTask_idx=priTask_idx)
            acc = test(student, args.device, test_loader, epoch)

            if acc > best_acc:
                best_acc = acc

                torch.save(expert.state_dict(), priTask_path)

        print("\nModel for %s Acc=%.2f%%" % (primitiveTask, best_acc*100))
