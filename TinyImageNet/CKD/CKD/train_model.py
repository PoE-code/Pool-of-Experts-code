from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
import network
from utils.dataloader import tinyimagenet_dataloader
from utils.TinyImageNet_hierarchy import TinyImageNet_Superclass
import os


def total_combine(Superclasses):
    All_class = []
    for cls_name in Superclasses:
        All_class += TinyImageNet_Superclass[cls_name]
        All_class.sort()

    return All_class


def distillation(y, teacher_scores, T):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    return l_kl


def train(args, teacher, student, device, train_loader, optimizer, epoch, total_idx):
    teacher.eval()
    student_EX, student_CL = student
    student_EX.eval()
    student_CL.train()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        t_logit = teacher(data)
        s_logit = student_CL(student_EX(data).detach())

        KL_loss = distillation(s_logit, t_logit[:, total_idx].detach(), 4)
        L1_loss = F.l1_loss(s_logit, t_logit[:, total_idx].detach())
        loss = KL_loss + (args.alpha * L1_loss)
        loss.backward()
        optimizer.step()

        if args.verbose and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLr: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))


def test(student, device, test_loader, cur_epoch, test_only=False):
    student_EX, student_CL = student
    student_EX.eval()
    student_CL.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = student_CL(student_EX(data))
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


def get_model(args):
    os.makedirs('DB_pretrained/CKD', exist_ok=True)

    total_idx = total_combine(args.Superclasses)

    train_loader, test_loader = tinyimagenet_dataloader(args, train_subidx=total_idx, test_subidx=total_idx)

    teacher = network.wresnet.wideresnet(depth=16, num_classes=200, widen_factor=10, dropRate=0.0)
    student_EX = network.wresnet.wideresnet_ex(depth=16, num_classes=200, widen_factor=2, dropRate=0.0)
    student_CL = network.wresnet.wideresnet_cl(depth=16, num_classes=len(total_idx), EX_widen_factor=2,
                                               widen_factor=(0.25*len(args.Superclasses)), dropRate=0.0)

    Oracle_path = './DB_pretrained/Oracle/Oracle_TinyImageNet.pt'
    EX_path = './DB_pretrained/Library/library_TinyImageNet.pt'
    CL_path = './DB_pretrained/CKD/CKD_CL_%s.pt' % args.Superclasses

    teacher.load_state_dict(torch.load(Oracle_path))
    student_EX.load_state_dict(torch.load(EX_path))
    teacher, student_EX = teacher.to(args.device), student_EX.to(args.device)
    teacher, student_EX = teacher.eval(), student_EX.eval()

    if args.model_pretrained is True:

        student_CL.load_state_dict(torch.load(CL_path))
        student_CL = student_CL.to(args.device)
        student_CL = student_CL.eval()

        student = [student_EX, student_CL]
        best_acc = test(student, args.device, test_loader, 0, True)
        print("\nModel for %s Acc=%.2f%%" % (args.Superclasses, best_acc*100))

        return

    student_CL = student_CL.to(args.device)

    optimizer_S = optim.SGD(list(student_EX.parameters())+list(student_CL.parameters()), lr=args.lr,
                            weight_decay=args.weight_decay, momentum=0.9)

    if args.scheduler:
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, [40, 80], 0.1)

    best_acc = 0

    student = [student_EX, student_CL]

    for epoch in range(1, args.model_epochs + 1):
        if args.scheduler:
            scheduler_S.step()

        train(args, teacher=teacher, student=student, device=args.device, train_loader=train_loader,
              optimizer=optimizer_S, epoch=epoch, total_idx=total_idx)
        acc = test(student, args.device, test_loader, epoch)

        if acc > best_acc:
            best_acc = acc

            torch.save(student_CL.state_dict(), CL_path)

    print("\nModel for %s Acc=%.2f%%" % (args.Superclasses, best_acc*100))
