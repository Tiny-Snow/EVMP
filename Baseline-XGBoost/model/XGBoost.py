# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- XGBoost
# Time:     2022.7.9

import numpy as np
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
        train_input = (train_input - np.mean(train_input) / np.std(train_input))
        log('>>>>> Size of training data: {}'.format(len(train_input)))
        log('>>>>> Size of training set: {}'.format(int(len(train_input) * (1 - cfg.val_ratio))))
        log('>>>>> Size of validation set: {}'.format(len(train_input) - int(len(train_input) * (1 - cfg.val_ratio))))
        x_train, x_test, y_train, y_test = train_test_split(train_input, train_value, test_size = cfg.val_ratio, random_state = 0, shuffle = True)
        model = xgboost.XGBRegressor()
        optimized_model = GridSearchCV(estimator = model, param_grid = cfg.param_grid, scoring = 'r2', cv = 5, verbose = 1, n_jobs = 4)
        optimized_model.fit(x_train, y_train)
        pred_train = optimized_model.predict(x_train)
        pred_test = optimized_model.predict(x_test)
        output(y_train, pred_train, y_test, pred_test, i)
        joblib.dump(optimized_model, cfg.save_root + '/XGBoost_{}.pkl'.format(i))


tot_train_MAE, tot_train_r2, tot_test_MAE, tot_test_r2 = 0, 0, 0, 0
data_tot = 0

def line(x, A, B):
    return A * x + B

def output(y_train, pred_train, y_test, pred_test, i, file_name = "XGBoost"):
    '''Output the prediction result of the i-th cross validation'''
    score_train_MAE = mean_absolute_error(y_train, pred_train)
    score_train_r2 = r2_score(y_train, pred_train)
    score_test_MAE = mean_absolute_error(y_test, pred_test)
    score_test_r2 = r2_score(y_test, pred_test)

    global tot_train_MAE, tot_train_r2, tot_test_MAE, tot_test_r2
    global data_tot

    tot_train_MAE += score_train_MAE
    tot_train_r2 += score_train_r2
    tot_test_MAE += score_test_MAE
    tot_test_r2 += score_test_r2
    data_tot += 1
    log("[{:03d}/{:03d}] Cross Validation".format(data_tot, cfg.cross_time))
    log("    train MAE: {:3.6f}, train R2: {:.2f}".format(score_train_MAE, score_train_r2))
    log("    valid MAE: {:3.6f}, valid R2: {:.2f}".format(score_test_MAE, score_test_r2))
    log("    mean train MAE: {:3.6f}, mean train R2: {:.2f}".format(tot_train_MAE / data_tot, tot_train_r2 / data_tot))
    log("    mean valid MAE: {:3.6f}, mean valid R2: {:.2f}".format(tot_test_MAE / data_tot, tot_test_r2 / data_tot))
    
    x = np.arange(min(y_train), max(y_train), 0.01)
    A1, B1 = optimize.curve_fit(line, y_test, pred_test)[0]
    y1 = A1 * x + B1
    
    plt.figure()
    plt.grid()
    plt.plot(x, y1, c = "red", linewidth = 3.0)
    plt.title(file_name)
    plt.scatter(y_train, pred_train, s = 3, c = "silver", marker = 'o', 
        label = "Train: mean $MAE$ = {:.2f}, mean $R^2$ = {:.2f}".format(tot_train_MAE / data_tot, tot_train_r2 / data_tot))
    plt.scatter(y_test, pred_test, s = 5, c = "darkgreen", marker = 'o', 
        label = "Valid: mean $MAE$ = {:.2f}, mean $R^2$ = {:.2f}".format(tot_test_MAE / data_tot, tot_test_r2 / data_tot))
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.xlabel('True Value')
    plt.ylabel('Predict Value')
    plt.legend()
    plt.savefig(cfg.fig_path + '/' + file_name + '_' + cfg.info + '-' + str(i) + '.pdf')