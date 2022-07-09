# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- XGBoost
# Time:     2022.7.9

import math
import random
import numpy as np
import csv

import config as cfg


def one_hot(index, length):
    '''
    return one-hot encoding, len(onehot) = length, onehot[index] = 1
    if index = length, then return [0, ..., 0]
    '''
    onehot = [0] * length
    if index != length:
        onehot[index] = 1
    return onehot

def onehot_encode(promoter):
    '''
    one-hot encoding for promoter of length `seq_len`
    one-hot encode element: B, A, T, C, G
    '''
    vocab = {'B': 0, 'A': 1, 'G': 2, 'C': 3, 'T': 4}
    vocab_size = len(vocab)
    onehots = []
    for i in range(cfg.seq_len):
        if i < len(promoter):
            onehots += one_hot(vocab[promoter[i]], vocab_size)
        else:
            onehots += one_hot(vocab_size, vocab_size)
    return np.array(onehots, dtype = np.float64)


def load_data():
    '''load train data and test data'''
    data, train_input, train_value = [], [], []
    with open(cfg.synthetic_data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        # load data
        for i, [ID,  mopromoter, promoter, act] in enumerate(reader):
            data.append([onehot_encode(promoter), math.log10(float(act))])
    random.shuffle(data)
    for promoter, act in data:
        train_input.append(promoter)
        train_value.append(act)
    return np.array(train_input), np.array(train_value)





