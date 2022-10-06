# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (XGBoost)
# Time:     2022.7.11

import random
import config as cfg
from model.EVMPPromoterEncoderFramework import run

if __name__ == '__main__':
    random.seed(20220719)
    run(cfg.param_xgb,  cfg.param_grid)