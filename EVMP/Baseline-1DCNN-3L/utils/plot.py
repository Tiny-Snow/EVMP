# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- 1D CNN (3 layers)
# Time:     2022.7.9

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import optimize
import matplotlib.pyplot as plt

import config as cfg

def line(x, A, B):
    return A * x + B

def output(y_train, pred_train, y_valid, pred_valid, y_test, pred_test, file_name):
    '''
    Output the prediction result of the train / valid /test
    
    Args:
        y_train         true value of train data
        pred_train      predict value of train data
        y_valid         true value of valid data
        pred_valid      predict value of valid data
        y_test          true value of test data
        pred_test       predict value of test data
        file_name       title of figure

    return:
        a dict, train/valid/test MAE and R2
    '''
    train_MAE   = mean_absolute_error(y_train, pred_train)
    train_r2    = r2_score(y_train, pred_train)
    valid_MAE   = mean_absolute_error(y_valid, pred_valid)
    valid_r2    = r2_score(y_valid, pred_valid)
    test_MAE    = mean_absolute_error(y_test, pred_test)
    test_r2     = r2_score(y_test, pred_test)
    
    x = np.arange(min(y_train), max(y_train), 0.01)
    A1, B1 = optimize.curve_fit(line, y_test, pred_test)[0]
    y1 = A1 * x + B1
    
    plt.figure()
    plt.grid()
    plt.plot(x, y1, c = "red", linewidth = 3.0)
    plt.title(file_name)
    plt.scatter(y_train, pred_train, s = 3, c = "silver", marker = 'o', 
        label = "Train: $MAE$ = {:.2f}, $R^2$ = {:.2f}".format(train_MAE, train_r2))
    plt.scatter(y_valid, pred_valid, s = 5, c = "y", marker = 'o', 
        label = "Valid: $MAE$ = {:.2f}, $R^2$ = {:.2f}".format(valid_MAE, valid_r2))
    plt.scatter(y_test, pred_test, s = 5, c = "darkgreen", marker = 'x', 
        label = " Test: $MAE$ = {:.2f}, $R^2$ = {:.2f}".format(test_MAE, test_r2))
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.xlabel('True Value')
    plt.ylabel('Predict Value')
    plt.legend()
    plt.savefig(cfg.save_fig)

    result = {
        'train': [train_MAE, train_r2], 
        'valid': [valid_MAE, valid_r2], 
        'test':  [test_MAE, test_r2]
    }
    return result
