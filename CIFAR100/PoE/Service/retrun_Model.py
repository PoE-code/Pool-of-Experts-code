from __future__ import print_function

import torch
import torch.nn.functional as F
import network
from utils.cifar100_hierarchy import CIFAR100_Superclass
from utils.dataloader import get_dataloader


def idx_search(User_select):
    All_class = []
    for cls_name in User_select:
        All_class += CIFAR100_Superclass[cls_name]
        All_class.sort()
    idx_dict = {}
    for supclass in User_select:
        idx_temp = []
        for i in CIFAR100_Superclass[supclass]:
            idx_temp.append(All_class.index(i))
        idx_dict[supclass] = idx_temp

    return idx_dict


def total_combine(User_select):
    All_class = []
    for cls_name in User_select:
        All_class += CIFAR100_Superclass[cls_name]
        All_class.sort()

    return All_class


def test(model_MQ, device, test_loader, idx_dict, queriedTask):

    model_MQ.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, logit_list = model_MQ(data, logits=True)

            for i, j in enumerate(logit_list):
                output[:, idx_dict[queriedTask[i]]] = j

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return correct / len(test_loader.dataset)


def get_MQ(args):

    total_idx = total_combine(args.queriedTask)
    idx_dict = idx_search(args.queriedTask)

    _, test_loader = get_dataloader(args, test_subidx=total_idx)

    library = network.wresnet.wideresnet_ex(depth=16, num_classes=args.Oracle_classes, widen_factor=1, dropRate=0.0)
    library.load_state_dict(torch.load('./DB_Pool of Experts/Library/library_cifar100.pt'))
    library = library.to(args.device)
    library.eval()

    experts = []
    for primitiveTask in args.queriedTask:
        priTask_idx = CIFAR100_Superclass[primitiveTask]
        priTask_path = './DB_Pool of Experts/Experts/expert_%s.pt' % primitiveTask
        expert = network.wresnet.wideresnet_cl(depth=16, num_classes=len(priTask_idx),
                                               EX_widen_factor=1, widen_factor=0.25, dropRate=0.0)
        expert.load_state_dict(torch.load(priTask_path))
        expert = expert.to(args.device)
        expert.eval()

        experts.append(expert)

    model_MQ = network.wresnet.wideresnet_MQ(library=library, experts=experts)

    best_acc = test(model_MQ, args.device, test_loader, idx_dict, args.queriedTask)
    print("\nModel_MQ  Acc=%.2f%%" % (best_acc*100))

    return model_MQ
