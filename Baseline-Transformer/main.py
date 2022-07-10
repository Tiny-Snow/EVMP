# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  EVMP BASELINES -- Transformer
# Time:     2022.7.9

import config as cfg
from utils.data_loader import load_data
from utils.helper import log
from model.BaselineTransformer import BaselineTransformer, run


if __name__ == '__main__':
    # load data
    log('>>>>> Loading data ...')
    train_loader, val_loader, test_loader, predict_loader = load_data()
    log('>>>>> Size of training data: {}'.format(len(train_loader) * cfg.batch_size + len(val_loader)))
    log('>>>>> Size of training set: {}'.format(len(train_loader) * cfg.batch_size))
    log('>>>>> Size of validation set: {}'.format(len(val_loader) * 1))
    if test_loader:
        log('>>>>> Size of test set: {}'.format(len(test_loader) * 1))
    else:
        log('>>>>> Not use test set')
    if predict_loader:
        log('>>>>> Size of predict set: {}'.format(len(predict_loader) * 1))
    else:
        log('>>>>> Not use predict set')

    # train
    run(train_loader, val_loader, test_loader, predict_loader)
