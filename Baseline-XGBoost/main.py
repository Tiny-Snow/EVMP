# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- XGBoost
# Time:     2022.7.9

import config as cfg
from model.XGBoost import run

if __name__ == '__main__':
    run(cfg.param_xgb,  cfg.param_grid)