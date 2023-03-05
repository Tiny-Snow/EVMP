# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- XGBoost
# Time:     2022.7.9


import math
import random
import matplotlib.pyplot as plt

import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import xgboost
from scipy import optimize

import config as cfg
from utils.data_loader import load_data
from utils.helper import log
from utils.plot import *


def run(x_train, y_train, x_valid, y_valid, x_test, y_test, param_xgb, param_grid, idx = 0):
    '''
    Train the model and test the data
    params:
        param_xgb           params of XGBRegressor
        param_grid          params of GridSearchCV
    '''
    # model
    model = xgboost.XGBRegressor(**param_xgb)
    optimized_model = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'r2', cv = 5, verbose = 1, n_jobs = 4)
    optimized_model.fit(x_train, y_train)
    # fit
    model.fit(x_train, y_train)
    pred_train = model.predict(x_train)
    pred_valid = model.predict(x_valid)
    pred_test = model.predict(x_test)
    # plot and save
    result = output(y_train, pred_train, y_valid, pred_valid, y_test, pred_test, file_name = 'XGBoost', idx = idx)
    joblib.dump(model, cfg.save_root + '/model_{}_C{}.pkl'.format(cfg.info, idx))

    log("Final Model #{}: ".format(idx))
    log("    train MAE: {:3.6f}, train R2: {:.2f}".format(result['train'][0], result['train'][1]))
    log("    valid MAE: {:3.6f}, valid R2: {:.2f}".format(result['valid'][0], result['valid'][1]))
    log("    test  MAE: {:3.6f}, test  R2: {:.2f}".format(result['test'][0], result['test'][1]))
    
    return result['train'][0], result['train'][1], result['valid'][0], result['valid'][1], result['test'][0], result['test'][1]
