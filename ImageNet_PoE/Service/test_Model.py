from __future__ import print_function

import torch
import torch.nn.functional as F
import network
from utils.ImageNet_hierarchy import ImageNet_Superclass
from utils.dataloader import ImageNet_dataloader


def idx_search(User_select):
    All_class = []
    for cls_name in User_select:
        All_class += ImageNet_Superclass[cls_name]
        All_class.sort()
    idx_dict = {}
    for supclass in User_select:
        idx_temp = []
        for i in ImageNet_Superclass[supclass]:
            idx_temp.append(All_class.index(i))
        idx_dict[supclass] = idx_temp

    return idx_dict


def total_combine(User_select):
    All_class = []
    for cls_name in User_select:
        All_class += ImageNet_Superclass[cls_name]
        All_class.sort()

    return All_class


def test(model_MQ, device, test_loader, idx_dict, queriedTask):

    library, expert = model_MQ
    library.eval()
    cl = []

    for i in expert:
        cl.append(i.eval())
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output_EX = library(data)

            logit_list = []

            for i in range(len(queriedTask)):
                logit_list.append(cl[i](output_EX))

            output = torch.cat(logit_list, dim=1)

            for i, j in enumerate(logit_list):
                output[:, idx_dict[queriedTask[i]]] = j

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    return correct / len(test_loader.dataset)


def test_MQ(args):

    total_idx = total_combine(args.queriedTask)
    idx_dict = idx_search(args.queriedTask)

    _, test_loader = ImageNet_dataloader(args, test_subidx=total_idx)

    library = network.pytorch_mobilenetV2.mobilenet_ex()
    library = library.to(args.device)
    library.eval()

    experts = []
    for primitiveTask in args.queriedTask:
        priTask_idx = ImageNet_Superclass[primitiveTask]
        priTask_path = './DB_Pool of Experts/Experts/expert_%s.pt' % primitiveTask
        expert = network.pytorch_mobilenetV2.mobilenet_cl(outchannel=40, num_classes=len(priTask_idx))
        expert = torch.load(priTask_path)
        expert = expert.to(args.device)
        expert.eval()

        experts.append(expert)

    best_acc = test((library, experts), args.device, test_loader, idx_dict, args.queriedTask)
    print("\nModel_MQ  Acc=%.2f%%" % (best_acc*100))

    return
