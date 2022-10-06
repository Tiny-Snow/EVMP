# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- SVM
# Time:     2022.7.9

import random
import config as cfg
from model.BaselineSVM import run

if __name__ == '__main__':
    random.seed(20220719)
    run()