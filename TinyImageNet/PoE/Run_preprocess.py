from __future__ import print_function
import argparse
import torch
from Preprocessing.train_Oracle import get_Oracle
from Preprocessing.train_library import get_library
from Preprocessing.train_experts import get_experts
from utils.envset import set_seed


def main():
    # Preprocessing settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--Oracle_classes', type=int, default=200, required=False)
    parser.add_argument('--Oracle_epochs', type=int, default=200, metavar='N', required=False)
    parser.add_argument('--Oracle_pretrained', default=True, required=False)
    parser.add_argument('--Oracle_step_size', type=int, default=80, metavar='S')

    parser.add_argument('--Library_pretrained', default=True, required=False)
    parser.add_argument('--Experts_epochs', type=int, default=100, metavar='N', required=False)
    parser.add_argument('--Experts_pretrained', default=True, required=False)
    parser.add_argument('--PoE_step_size', type=int, default=40, metavar='S')

    parser.add_argument('--device', type=str, default='cpu', required=False)
    parser.add_argument('--batch_size', type=int, default=256, metavar='N')
    parser.add_argument('--alpha', type=int, default=0.3)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N', required=False,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--verbose', default=True, required=False)
    parser.add_argument('--scheduler', default=True, required=False)

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    set_seed(args)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.device = torch.device("cuda" if use_cuda else "cpu")

    print(args)

    # get Oracle model
    Oracle = get_Oracle(args)

    # Construct PoE
    library = get_library(args, Oracle)
    get_experts(args, Oracle, library)


if __name__ == '__main__':
    main()
