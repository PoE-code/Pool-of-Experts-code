from __future__ import print_function
import argparse
import torch
from utils.envset import set_seed
from Service.retrun_Model import get_MQ
from utils.cout_flops import total_flops


def main():
    # Service settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--Oracle_classes', type=int, default=100, required=False)
    parser.add_argument('--device', type=str, default='cpu', required=False)
    parser.add_argument('--batch_size', type=int, default=256, metavar='N')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--queriedTask', required=True, nargs='+', type=str)

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    set_seed(args)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.device = torch.device("cuda" if use_cuda else "cpu")

    print(args)

    # get specialised Model
    print('\nQueried Task:', args.queriedTask)
    model_MQ = get_MQ(args)
    print('  + Number of FLOPs: %.2fB' % (total_flops(model_MQ) / 1e9))


if __name__ == '__main__':
    main()
