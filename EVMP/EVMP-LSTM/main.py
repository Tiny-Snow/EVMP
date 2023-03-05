# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (LSTM)
# Time:     2022.5.7

import config as cfg
import numpy as np

from utils.data_loader import load_data, train_valid_spilt
from utils.helper import log, create_dir, seed_all
from model.EVMPPromoterEncoderFramework import EVMPPromoterEncoderFramework, run


if __name__ == '__main__':
    seed_all(cfg.seed)

    create_dir('./output')
    create_dir('./save')

    # load data
    log('>>>>> Loading data ...')
    train_data_loader, test_loader, predict_loader = load_data()

    metric = np.zeros((6))

    # cross validation
    for run_num in range(cfg.fold):
        log('>>>>> Run #{} ...'.format(run_num))
        train_loader, val_loader = train_valid_spilt(train_data_loader, cfg.val_ratio)
        log('>>>>> Size of training set: {}'.format(len(train_loader.dataset)))
        log('>>>>> Size of validation set: {}'.format(len(val_loader.dataset)))
        log('>>>>> Size of test set: {}'.format(len(test_loader.dataset)))

        if predict_loader:
            log('>>>>> Size of predict set: {}'.format(len(predict_loader.dataset)))
        else:
            log('>>>>> Not use predict set')

        # train
        train_MAE, train_R2, valid_MAE, valid_R2, test_MAE, test_R2 = \
            run(train_loader, val_loader, test_loader, predict_loader, idx = run_num)
        metric += np.array([train_MAE, train_R2, valid_MAE, valid_R2, test_MAE, test_R2])

        log('----------------------------------------------------------------------')

    # print mean metric
    metric /= cfg.fold
    log('>>>>> Mean metric:')
    log('>>>>> Train MAE: {:.4f}, Train R2: {:.4f}'.format(metric[0], metric[1]))
    log('>>>>> Valid MAE: {:.4f}, Valid R2: {:.4f}'.format(metric[2], metric[3]))
    log('>>>>> Test MAE: {:.4f}, Test R2: {:.4f}'.format(metric[4], metric[5]))
