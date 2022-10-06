# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- 1D CNN (50 layers)
# Time:     2022.7.9

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import random
import math

import config as cfg


class PromoterDataset(Dataset):
    '''
    Promoter Dataset
    
    Args:
        bases               array of promoter (one-hot encode)
        acts                promoter fluorescence intensity

    Dataset item:
        with act            (base, act)
        no act              base
    '''
    def __init__(self, bases, acts = None):
        super(Dataset, self).__init__()
        self.bases = bases
        self.acts = acts

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, idx):
        base = self.bases[idx]
        promoter = np.array(base, dtype = np.float64)
        if self.acts:
            act = self.acts[idx]
            return (promoter, act)
        else:
            return promoter


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
    val_bases, val_acts     = [], []
    test_bases, test_acts   = [], []

    # [test, train, valid]
    val_percent = int(len(synthetic_promoter) * (1 - cfg.val_ratio)) // cfg.batch_size * cfg.batch_size
    test_percent = int(len(synthetic_promoter) * (cfg.test_ratio)) // cfg.batch_size * cfg.batch_size

    for p in wilds:
        train_bases.append(onehot_encode(p['promoter']))
        train_acts.append(p['act'])

    for p in synthetic_promoter[: test_percent]:
        test_bases.append(onehot_encode(p['promoter']))
        test_acts.append(p['act'])
    for p in synthetic_promoter[test_percent: val_percent]:
        train_bases.append(onehot_encode(p['promoter']))
        train_acts.append(p['act'])
    for p in synthetic_promoter[val_percent: ]:
        val_bases.append(onehot_encode(p['promoter']))
        val_acts.append(p['act'])

    train_set       = PromoterDataset(train_bases, train_acts)
    train_loader    = DataLoader(train_set, batch_size = cfg.batch_size, shuffle = True, drop_last = False)
    val_set         = PromoterDataset(val_bases, val_acts)
    val_loader      = DataLoader(val_set, batch_size = 1, shuffle = False, drop_last = False)
    test_set        = PromoterDataset(test_bases, test_acts)
    test_loader     = DataLoader(test_set, batch_size = 1, shuffle = False, drop_last = False)

    # load predict data
    if cfg.if_predict:
        predict_promoter = []
        with open(cfg.predict_data_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for i, [ID, mopromoter, promoter] in enumerate(reader):
                if mopromoter in wild_promoter:
                    predict_promoter.append({'Mopromoter': mopromoter, 'promoter': promoter})
                else:
                    predict_promoter.append({'Mopromoter': mopromoter, 'promoter': promoter})
        predict_bases = []
        for p in predict_promoter:
            predict_bases.append(onehot_encode(p['promoter']))
        predict_set = PromoterDataset(predict_bases)
        predict_loader = DataLoader(predict_set, batch_size = 1, shuffle = False, drop_last = False)
    else:
        predict_loader = None

    return train_loader, val_loader, test_loader, predict_loader
