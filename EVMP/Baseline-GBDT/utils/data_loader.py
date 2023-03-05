# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- GBDT
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
    '''
    Load train/val/test/predict data
    '''
    # load train/val data
    wild_promoter = {}
    synthetic_promoter = []
    with open(cfg.wild_data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, [ID, mopromoter, promoter, act] in enumerate(reader):
            wild_promoter[mopromoter] = {'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))}
    with open(cfg.synthetic_data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, [ID, mopromoter, promoter, act] in enumerate(reader):
            synthetic_promoter.append({'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))})

    wilds = list(wild_promoter.values())
    random.shuffle(wilds)
    random.shuffle(synthetic_promoter)
    
    train_bases, train_acts = [], []
    test_bases, test_acts   = [], []

    # [test, train, valid]
    test_percent = int(len(synthetic_promoter) * cfg.test_ratio)

    for p in wilds:
        train_bases.append(onehot_encode(p['promoter']))
        train_acts.append(p['act'])

    for p in synthetic_promoter[: test_percent]:
        test_bases.append(onehot_encode(p['promoter']))
        test_acts.append(p['act'])
    for p in synthetic_promoter[test_percent: ]:
        train_bases.append(onehot_encode(p['promoter']))
        train_acts.append(p['act'])

    return train_bases, train_acts, test_bases, test_acts

    # TODO: load predict data


