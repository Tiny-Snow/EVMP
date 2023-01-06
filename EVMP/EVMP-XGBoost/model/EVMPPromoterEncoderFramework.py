# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (XGBoost)
# Time:     2022.7.11

import math
import random
import matplotlib.pyplot as plt

import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost
from scipy import optimize

import config as cfg
from utils.data_loader import load_data
from utils.helper import log
from utils.plot import *


def run(param_xgb, param_grid):
    '''
    Train the model and test the data
    params:
        param_xgb           params of XGBRegressor
        param_grid          params of GridSearchCV
    '''
    # cross validation
    for i in range(1, cfg.cross_time + 1):
        # load data
        log('>>>>> Loading data ...')
        train_input, train_value = load_data()
        x_train, x_valid, y_train, y_valid = train_test_split(train_input, train_value, test_size = cfg.val_ratio, random_state = 0, shuffle = True)
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = cfg.test_ratio, random_state = 0, shuffle = True)
        log('>>>>> Size of training set: {}'.format(len(x_train)))
        log('>>>>> Size of validation set: {}'.format(len(x_valid)))
        log('>>>>> Size of test set: {}'.format(len(x_test)))

        model = xgboost.XGBRegressor()
        optimized_model = GridSearchCV(estimator = model, param_grid = cfg.param_grid, scoring = 'r2', cv = 5, verbose = 1, n_jobs = 4)
        optimized_model.fit(x_train, y_train)

        model.fit(x_train, y_train)
        pred_train = model.predict(x_train)
        pred_valid = model.predict(x_valid)
        pred_test = model.predict(x_test)
        
        logs = output(y_train, pred_train, y_valid, pred_valid, y_test, pred_test, file_name = 'EVMP-XGBoost', i = i)
        for s in logs:
            log(s)

        joblib.dump(model, cfg.save_root + '/EVMP-XGBoost_{}.pkl'.format(i))