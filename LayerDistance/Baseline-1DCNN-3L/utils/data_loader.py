# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- 1D CNN (Layer 3)
# Time:     2022.7.9

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import random
import math

import config as cfg


def one_hot(index, length):
    '''
    return one-hot encoding, len(onehot) = length, onehot[index] = 1
    if index = length, then return [0, ..., 0]
    '''
    onehot = [0] * length
    if index != length:
        onehot[index] = 1
    return np.array(onehot)

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
            onehots.append(one_hot(vocab[promoter[i]], vocab_size))
        else:
            onehots.append(one_hot(vocab_size, vocab_size))
    return np.array(onehots, dtype = np.float64)


def load_data():
    '''
    Load train/val/test/predict data
    '''
    # load train data and test data
    synthetic_data, base_data = [], []
    for dataset_pair in [(cfg.synthetic_data_path, synthetic_data), (cfg.base_promoter_path, base_data)]:
        dataset, arr = dataset_pair[0], dataset_pair[1]
        with open(dataset, 'r') as f:
            train_data = csv.reader(f)
            next(train_data)
            # load data
            for i, [ID,  mopromoter, promoter, act] in enumerate(train_data):
                arr.append([onehot_encode(promoter), math.log10(float(act))])
    
    # split data into train/val
    random.shuffle(synthetic_data)
    train_data = synthetic_data[int(len(synthetic_data) * cfg.val_ratio): ]
    val_data = synthetic_data[: int(len(synthetic_data) * cfg.val_ratio)]
    train_loader = DataLoader(train_data, batch_size = cfg.batch_size, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = 1, shuffle = False)
    base_loader = DataLoader(base_data, batch_size = 1, shuffle = False)

    return train_loader, val_loader, None, None, base_loader
