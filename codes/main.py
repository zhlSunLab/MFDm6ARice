#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : MFDm6ARice
@File        : main.py
@Author      : Mengya Liu
@Date        : 2024/5/9 18:33
@Description : Rice m6A prediction

"""
print(__doc__)

import os
import sys
import numpy as np
import torch
from train_CV import crossCV
from prediction import pred
from param_options import *


def main():

    outpath = opt.outpath
    inputpath = opt.inputpath
    exp_name = opt.exp_name

    if not os.path.exists(outpath):
        print("The output path not exist! Create a new folder...\n")
        os.makedirs(outpath)
    if not os.path.exists(inputpath):
        print("The input data not exist! Error\n")
        sys.exit()

    if exp_name.startswith('cross'):
        crossCV(opt)
    elif exp_name.startswith('pred'):
        pred(opt)


if __name__ == "__main__":

    opt = getArg()
    getDefaultPara(opt)

    seed = opt.seed

    # use cuda or cpu
    if opt.device.startswith('cuda') & (torch.cuda.is_available()):
        opt.cudaFlag = True
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        opt.cudaFlag = False
        opt.device = 'cpu'
        np.random.seed(seed)
        torch.manual_seed(seed)

    main()

