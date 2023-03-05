# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (XGBoost)
# Time:     2022.7.11

import random
import config as cfg
import numpy as np

from utils.data_loader import load_data
from utils.helper import log, create_dir, seed_all
from sklearn.model_selection import train_test_split
from model.EVMPPromoterEncoderFramework import run


if __name__ == '__main__':
    seed_all(cfg.seed)

    create_dir('./output')
    create_dir('./save')

    # load data
    log('>>>>> Loading data ...')
    train_input, train_value, x_test, y_test = load_data()

    metric = np.zeros((6))

    # cross validation
    for run_num in range(cfg.fold):
        log('>>>>> Run #{} ...'.format(run_num))

        train_data = np.concatenate((np.array(train_input), np.array(train_value).reshape(-1, 1)), axis = 1)
        np.random.shuffle(train_data)
        train_input = train_data[:, :-1]
        train_value = train_data[:, -1]

        x_train, x_valid, y_train, y_valid = train_test_split(train_input, train_value, test_size = cfg.val_ratio, random_state = cfg.seed, shuffle = True)

        log('>>>>> Size of training set: {}'.format(len(x_train)))
        log('>>>>> Size of validation set: {}'.format(len(x_valid)))
        log('>>>>> Size of test set: {}'.format(len(x_test)))

        # train
        train_MAE, train_R2, valid_MAE, valid_R2, test_MAE, test_R2 = \
            run(x_train, y_train, x_valid, y_valid, x_test, y_test, param_xgb = cfg.param_xgb, param_grid = cfg.param_grid, idx = run_num)
        metric += np.array([train_MAE, train_R2, valid_MAE, valid_R2, test_MAE, test_R2])

        log('----------------------------------------------------------------------')

    # print mean metric
    metric /= cfg.fold
    log('>>>>> Mean metric:')
    log('>>>>> Train MAE: {:.4f}, Train R2: {:.4f}'.format(metric[0], metric[1]))
    log('>>>>> Valid MAE: {:.4f}, Valid R2: {:.4f}'.format(metric[2], metric[3]))
    log('>>>>> Test MAE: {:.4f}, Test R2: {:.4f}'.format(metric[4], metric[5]))
