# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- RF
# Time:     2022.7.9

import random
import config as cfg
from model.BaselineRamdomForest import run

if __name__ == '__main__':
    random.seed(20220719)
    run()