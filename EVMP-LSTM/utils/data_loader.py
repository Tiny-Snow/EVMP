# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (LSTM)
# Time:     2022.5.7

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import csv
import random
import math

import config as cfg


class VarPromoterDataset(Dataset):
    '''
        Variation-based Promoter Dataset

        bases               array of base promoter (one-hot encode)
        vars                index of variations
        acts                promoter fluorescence intensity

        Dataset item:
        with act:           ({'base': base, 'var': var}, act)
        no act:             {'base': base, 'var': var}
    '''
    def __init__(self, bases, vars, acts = None):
        super(Dataset, self).__init__()
        self.bases = bases
        self.vars = vars
        self.acts = acts

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, index):
        base = self.bases[index]
        var = self.vars[index]
        if self.acts:
            act = self.acts[index]
            promoter = {'base': np.array(base, dtype = np.float64), 'var': np.array(var, dtype = np.float64)}
            return (promoter, act)
        else:
            promoter = {'base': np.array(base, dtype = np.float64), 'var': np.array(var, dtype = np.float64)}
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
    onehots = []
    for i in range(cfg.seq_len):
        onehot = []
        if i < len(promoter):
            onehots.append(one_hot(vocab[promoter[i]], 5))
        else:
            onehots.append(one_hot(5, 5))
    return np.array(onehots)


def var_encode(wild_index, mother_promoter, origin_promoter):
    '''
    variation encoding for origin promoter and mother promoter (wild_index th wild promoter).
    if number of variations > `max variation`, then return None
    if origin promoter = mother promoter, then return [0, ..., 0]
    '''
    vocab = {'B': 0, 'A': 1, 'G': 2, 'C': 3, 'T': 4}
    vars = []
    origin_promoter += 'B' * (cfg.seq_len - len(origin_promoter))
    mother_promoter += 'B' * (cfg.seq_len - len(mother_promoter))
    for i in range(cfg.seq_len):
        if origin_promoter[i] != mother_promoter[i]:
            diff_mer = [vocab[origin_promoter[i]] if k >= 0 and k < cfg.seq_len else 0 
                            for k in range(i - ((cfg.mer - 1) // 2), i + (cfg.mer - ((cfg.mer - 1) // 2)))]
            encode = wild_index * (cfg.seq_len * (5 ** cfg.mer) + 1) +  i * (5 ** cfg.mer)
            for k in range(cfg.mer):
                encode += (5 ** k) * diff_mer[k]
            vars.append(encode + 1)
    if len(vars) <= cfg.max_variation:
        vars += [0] * (cfg.max_variation - len(vars))
        return np.array(vars)
    return None


def load_data():
    '''
    Load train/val/test/predict data
    '''
    # laod train data
    wild_promoter = {}
    synthetic_promoter = []
    with open(cfg.wild_data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, [ID, mopromoter, promoter, act] in enumerate(reader):
            wild_promoter[mopromoter] = {'Moindex': i,'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))}
    with open(cfg.synthetic_data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, [ID, mopromoter, promoter, act] in enumerate(reader):
            synthetic_promoter.append({'Moindex': wild_promoter[mopromoter]['Moindex'], 'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))})

    percent = int(len(synthetic_promoter) * (1 - cfg.val_ratio)) // cfg.batch_size * cfg.batch_size - 1
    random.shuffle(synthetic_promoter)
    train_bases, train_vars, train_acts = [], [], []
    val_bases, val_vars, val_acts = [], [], []
    
    for _, p in wild_promoter.items():
        train_bases.append(onehot_encode(p['Mopromoter']))
        train_vars.append(var_encode(p['Moindex'], p['Mopromoter'], p['promoter']))
        train_acts.append(p['act'])
    for p in synthetic_promoter[: percent]:
        var_encoding = var_encode(p['Moindex'], p['Mopromoter'], p['promoter'])
        if not var_encoding is None:
            train_bases.append(onehot_encode(p['Mopromoter']))
            train_vars.append(var_encoding)
            train_acts.append(p['act'])
    for p in synthetic_promoter[percent: ]:
        var_encoding = var_encode(p['Moindex'], p['Mopromoter'], p['promoter'])
        if not var_encoding is None:
            val_bases.append(onehot_encode(p['Mopromoter']))
            val_vars.append(var_encoding)
            val_acts.append(p['act'])

    train_set = VarPromoterDataset(train_bases, train_vars, train_acts)
    train_loader = DataLoader(train_set, batch_size = cfg.batch_size, shuffle = True, drop_last = True)
    val_set = VarPromoterDataset(val_bases, val_vars, val_acts)
    val_loader = DataLoader(val_set, batch_size = 1, shuffle = False, drop_last = True)

    # load test data
    if cfg.have_test:
        test_promoter = []
        with open(cfg.test_data_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for i, [ID, mopromoter, promoter, act] in enumerate(reader):
                if mopromoter in wild_promoter:
                    test_promoter.append({'Moindex': wild_promoter[mopromoter]['Moindex'], 'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))})
                else:
                    test_promoter.append({'Moindex': float('inf'), 'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))})
        test_bases, test_vars, test_acts = [], [], []
        for p in test_promoter:
            var_encoding = var_encode(p['Moindex'], p['Mopromoter'], p['promoter'])
            if not var_encoding is None:
                test_bases.append(onehot_encode(p['Mopromoter']))
                test_vars.append(var_encoding)
                test_acts.append(p['act'])
        test_set = VarPromoterDataset(test_bases, test_vars, test_acts)
        test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, drop_last = True)
    else:
        test_loader = None

    # load predict data
    if cfg.if_predict:
        predict_promoter = []
        with open(cfg.predict_data_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for i, [ID, mopromoter, promoter, act] in enumerate(reader):
                if mopromoter in wild_promoter:
                    predict_promoter.append({'Moindex': wild_promoter[mopromoter]['Moindex'], 'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))})
                else:
                    predict_promoter.append({'Moindex': float('inf'), 'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))})
        predict_bases, predict_vars = [], []
        for p in predict_promoter:
            var_encoding = var_encode(p['Moindex'], p['Mopromoter'], p['promoter'])
            if not var_encoding is None:
                predict_bases.append(onehot_encode(p['Mopromoter']))
                predict_vars.append(var_encoding)
        predict_set = VarPromoterDataset(predict_bases, predict_vars)
        predict_loader = DataLoader(predict_set, batch_size = 1, shuffle = False, drop_last = True)
    else:
        predict_loader = None

    return train_loader, val_loader, test_loader, predict_loader

