# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- GBDT
# Time:     2022.7.9

import math
import random
import matplotlib.pyplot as plt

import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from scipy import optimize

import config as cfg
from utils.data_loader import load_data
from utils.helper import log
from utils.plot import *


def run():
    '''
    Train the model and test the data
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

        model = GradientBoostingRegressor(
            n_estimators = 200,     # number of trees 
            max_depth = 5,          # max depth of tree
            learning_rate = 0.1,
        )

        model.fit(x_train, y_train)
        pred_train = model.predict(x_train)
        pred_valid = model.predict(x_valid)
        pred_test = model.predict(x_test)
        
        logs = output(y_train, pred_train, y_valid, pred_valid, y_test, pred_test, file_name = 'GBDT', i = i)
        for s in logs:
            log(s)

        joblib.dump(model, cfg.save_root + '/GBDT_{}.pkl'.format(i))

