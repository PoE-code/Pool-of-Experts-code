from __future__ import print_function

import network


def get_library(args):
    student_library = network.pytorch_mobilenetV2.mobilenet_ex()
    student_library = student_library.to(args.device)
    student_library.eval()

    return student_library
