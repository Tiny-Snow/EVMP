# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- SVM
# Time:     2022.7.9

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import optimize
import matplotlib.pyplot as plt

import config as cfg


tot_train_MAE, tot_train_r2, tot_valid_MAE, tot_valid_r2, tot_test_MAE, tot_test_r2 = [0] * 6
data_tot = 0


def line(x, A, B):
    return A * x + B

def output(y_train, pred_train, y_valid, pred_valid, y_test, pred_test, file_name, i):
    '''
    Output the prediction result of the i-th cross validation
    
    Args:
        y_train         true value of train data
        pred_train      predict value of train data
        y_valid         true value of valid data
        pred_valid      predict value of valid data
        y_test          true value of test data
        pred_test       predict value of test data
        file_name       title of figure

    return:
        logs            an array of log infos

    '''
    global tot_train_MAE, tot_train_r2, tot_valid_MAE, tot_valid_r2, tot_test_MAE, tot_test_r2
    global data_tot

    train_MAE   = mean_absolute_error(y_train, pred_train)
    train_r2    = r2_score(y_train, pred_train)
    valid_MAE   = mean_absolute_error(y_valid, pred_valid)
    valid_r2    = r2_score(y_valid, pred_valid)
    test_MAE    = mean_absolute_error(y_test, pred_test)
    test_r2     = r2_score(y_test, pred_test)

    tot_train_MAE   += train_MAE
    tot_train_r2    += train_r2
    tot_valid_MAE   += valid_MAE
    tot_valid_r2    += valid_r2
    tot_test_MAE    += test_MAE
    tot_test_r2     += test_r2

    data_tot        += 1

    logs = []
    logs.append("[{:03d}/{:03d}] Cross Validation".format(data_tot, cfg.cross_time))
    logs.append("    train MAE: {:3.6f}, train R2: {:.2f}".format(train_MAE, train_r2))
    logs.append("    valid MAE: {:3.6f}, valid R2: {:.2f}".format(valid_MAE, valid_r2))
    logs.append("     test MAE: {:3.6f},  test R2: {:.2f}".format(test_MAE, test_r2))
    logs.append("    mean train MAE: {:3.6f}, mean train R2: {:.2f}".format(tot_train_MAE / data_tot, tot_train_r2 / data_tot))
    logs.append("    mean valid MAE: {:3.6f}, mean valid R2: {:.2f}".format(tot_valid_MAE / data_tot, tot_valid_r2 / data_tot))
    logs.append("    mean  test MAE: {:3.6f}, mean  test R2: {:.2f}".format(tot_test_MAE / data_tot, tot_test_r2 / data_tot))

    x = np.arange(min(y_train), max(y_train), 0.01)
    A1, B1 = optimize.curve_fit(line, y_test, pred_test)[0]
    y1 = A1 * x + B1
    
    plt.figure()
    plt.grid()
    plt.plot(x, y1, c = "red", linewidth = 3.0)
    plt.title(file_name)
    plt.scatter(y_train, pred_train, s = 3, c = "silver", marker = 'o', 
        label = "Train: mean $MAE$ = {:.2f}, mean $R^2$ = {:.2f}".format(tot_train_MAE / data_tot, tot_train_r2 / data_tot))
    plt.scatter(y_valid, pred_valid, s = 5, c = "y", marker = 'o', 
        label = "Valid: mean $MAE$ = {:.2f}, mean $R^2$ = {:.2f}".format(tot_valid_MAE / data_tot, tot_valid_r2 / data_tot))
    plt.scatter(y_test, pred_test, s = 5, c = "darkgreen", marker = 'x', 
        label = " Test: mean $MAE$ = {:.2f}, mean $R^2$ = {:.2f}".format(tot_test_MAE / data_tot, tot_test_r2 / data_tot))
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.xlabel('True Value')
    plt.ylabel('Predict Value')
    plt.legend()
    plt.savefig(cfg.fig_path + '/' + file_name + '_' + cfg.info + '-' + str(i) + '.pdf')

    return logs

