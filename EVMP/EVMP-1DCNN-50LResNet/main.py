# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (1DCNN 50 layers ResNet)
# Time:     2022.7.12

import random

import config as cfg

from utils.data_loader import load_data
from utils.helper import log
from model.EVMPPromoterEncoderFramework import run


if __name__ == '__main__':
    random.seed(20220719)

    # load data
    log('>>>>> Loading data ...')
    train_loader, val_loader, test_loader, predict_loader = load_data()
    log('>>>>> Size of training set: {}'.format(len(train_loader) * cfg.batch_size))
    log('>>>>> Size of validation set: {}'.format(len(val_loader) * 1))
    log('>>>>> Size of test set: {}'.format(len(test_loader) * 1))
    if predict_loader:
        log('>>>>> Size of predict set: {}'.format(len(predict_loader) * 1))
    else:
        log('>>>>> Not use predict set')

    # train
    run(train_loader, val_loader, test_loader, predict_loader)
