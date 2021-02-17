from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.optim as optim
import network
from utils.dataloader import tinyimagenet_dataloader
from utils.TinyImageNet_hierarchy import TinyImageNet_Superclass
import os


def idx_search(Superclasses):
    All_class = []
    for cls_name in Superclasses:
        All_class += TinyImageNet_Superclass[cls_name]
        All_class.sort()
    idx_dict = {}
    for supclass in Superclasses:
        idx_temp = []
        for i in TinyImageNet_Superclass[supclass]:
            idx_temp.append(All_class.index(i))
        idx_dict[supclass] = idx_temp

    return idx_dict


def total_combine(Superclasses):
    All_class = []
    for cls_name in Superclasses:
        All_class += TinyImageNet_Superclass[cls_name]
        All_class.sort()

    return All_class


def SD_distillation(y, teacher_scores, T, idx):
    p = F.log_softmax(y/T, dim=1) # student 20
    q = F.softmax(teacher_scores/T, dim=1) # teacher 5
    q_zeros = torch.zeros_like(p)
    q_zeros[:, idx] = q
    l_kl = F.kl_div(p, q_zeros, size_average=False) * (T**2) / y.shape[0]
    return l_kl


def train(args, teacher_set, student, device, train_loader, optimizer, epoch, idx_dict):
    teacher_EX, teacher = teacher_set
    teacher_EX.eval()
    cl = []

    for i in teacher:
        cl.append(i.eval())

    student_EX, student_CL = student
    student_EX.train()
    student_CL.train()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        t_avmap = teacher_EX(data).detach()

        logit_list = []

        for i in range(len(args.Superclasses)):
            logit_list.append(cl[i](t_avmap).detach())

        s_logit = student_CL(student_EX(data))

        loss_list = []
        for i, j in enumerate(logit_list):
            temp_loss = SD_distillation(s_logit, j.detach(), 4, idx_dict[args.Superclasses[i]])
            loss_list.append(temp_loss)

        loss = sum(loss_list) / len(loss_list)
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
    os.makedirs('DB_pretrained/SD_CKD', exist_ok=True)

    total_idx = total_combine(args.Superclasses)
    idx_dict = idx_search(args.Superclasses)

    train_loader, test_loader = tinyimagenet_dataloader(args, train_subidx=total_idx, test_subidx=total_idx)

    student_EX = network.wresnet.wideresnet_ex(depth=16, num_classes=200, widen_factor=2, dropRate=0.0)
    student_CL = network.wresnet.wideresnet_cl(depth=16, num_classes=len(total_idx), EX_widen_factor=2,
                                               widen_factor=(0.25*len(args.Superclasses)), dropRate=0.0)

    student_EX_path = './DB_pretrained/SD_CKD/SD_CKD_EX_%s.pt' % args.Superclasses
    student_CL_path = './DB_pretrained/SD_CKD/SD_CKD_CL_%s.pt' % args.Superclasses

    if args.model_pretrained is True:
        student_EX.load_state_dict(torch.load(student_EX_path))
        student_CL.load_state_dict(torch.load(student_CL_path))

        student_EX, student_CL = student_EX.to(args.device), student_CL.to(args.device)
        student_EX, student_CL = student_EX.eval(), student_CL.eval()

        student = [student_EX, student_CL]
        best_acc = test(student, args.device, test_loader, 0, True)
        print("\nModel for %s Acc=%.2f%%" % (args.Superclasses, best_acc*100))

        return

    student_EX, student_CL = student_EX.to(args.device), student_CL.to(args.device)

    library_path = './DB_pretrained/Library/library_TinyImageNet.pt'

    library = network.wresnet.wideresnet_ex(depth=16, num_classes=200, widen_factor=2, dropRate=0.0)
    library.load_state_dict(torch.load(library_path))
    library = library.to(args.device)
    library.eval()

    expert = []
    for s in args.Superclasses:
        expert_path = './DB_pretrained/Experts/expert_%s.pt' % s
        e = network.wresnet.wideresnet_cl(depth=16, num_classes=len(TinyImageNet_Superclass[s]),
                                          EX_widen_factor=2, widen_factor=0.25, dropRate=0.0)
        e.load_state_dict(torch.load(expert_path))
        e = e.to(args.device)
        e.eval()
        expert.append(e)

    optimizer_S = optim.SGD(list(student_EX.parameters())+list(student_CL.parameters()), lr=args.lr,
                            weight_decay=args.weight_decay, momentum=0.9)

    if args.scheduler:
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, [80, 160], 0.1)

    best_acc = 0

    student = [student_EX, student_CL]

    for epoch in range(1, args.model_epochs + 1):
        if args.scheduler:
            scheduler_S.step()

        train(args, teacher_set=(library, expert), student=student, device=args.device, train_loader=train_loader,
              optimizer=optimizer_S, epoch=epoch, idx_dict=idx_dict)
        acc = test(student, args.device, test_loader, epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save(student_EX.state_dict(), student_EX_path)
            torch.save(student_CL.state_dict(), student_CL_path)

    print("\nModel for %s Acc=%.2f%%" % (args.Superclasses, best_acc*100))
