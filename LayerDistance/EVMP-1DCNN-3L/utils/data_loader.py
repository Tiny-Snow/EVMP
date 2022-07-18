# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (1DCNN 3 layers)
# Time:     2022.7.12

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
    
    Args:
        bases               array of base promoter (one-hot encode)
        vars                k-mer variant subsequence masked promoter
        acts                promoter fluorescence intensity

    Dataset item:
        with act            ({'base': base, 'var': var}, act)
        no act              {'base': base, 'var': var}
    '''
    def __init__(self, bases, vars, acts = None):
        super(Dataset, self).__init__()
        self.bases = bases
        self.vars = vars
        self.acts = acts

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, idx):
        base = self.bases[idx]
        var = self.vars[idx]
        promoter = {
            'base': np.array(base, dtype = np.float64), 
            'var': np.array(var, dtype = np.float64), 
        }
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
        onehot = []
        if i < len(promoter):
            onehots.append(one_hot(vocab[promoter[i]], vocab_size))
        else:
            onehots.append(one_hot(vocab_size, vocab_size))
    return np.array(onehots)


def var_mask(mother_promoter, origin_promoter):
    '''
    variation subsequence masked promoter for origin promoter and mother promoter (wild_index th wild promoter).
    if one pos not in any k-mer variation subsequence, then that pos is masked (all zero)
    
    return masked synthetic promoters
    '''
    vocab = {'B': 0, 'A': 1, 'G': 2, 'C': 3, 'T': 4}
    vocab_size = len(vocab)
    origin_promoter += 'B' * (cfg.seq_len - len(origin_promoter))
    mother_promoter += 'B' * (cfg.seq_len - len(mother_promoter))
    mask_promoter = ['Z'] * cfg.seq_len
    for i in range(cfg.seq_len):
        if origin_promoter[i] != mother_promoter[i]:
            for k in range(max(0, i - ((cfg.mer - 1) // 2)), min(i + (cfg.mer - ((cfg.mer - 1) // 2)), cfg.seq_len)):
                mask_promoter[k] = origin_promoter[k]
    mask = []
    for p in mask_promoter:
        if p == 'Z':
            mask.append(one_hot(vocab_size, vocab_size))
        else:
            mask.append(one_hot(vocab[p], vocab_size))
    return np.array(mask)


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

    synthetic_percent = int(len(synthetic_promoter) * (1 - cfg.val_ratio)) // cfg.batch_size * cfg.batch_size
    wilds = list(wild_promoter.values())
    random.shuffle(wilds)
    random.shuffle(synthetic_promoter)
    train_bases, train_vars, train_acts = [], [], []
    val_bases, val_vars, val_acts = [], [], []

    for p in wilds:
        train_bases.append(onehot_encode(p['Mopromoter']))
        train_vars.append(var_mask(p['Mopromoter'], p['promoter']))
        train_acts.append(p['act'])
    for p in synthetic_promoter[: synthetic_percent]:
        train_bases.append(onehot_encode(p['Mopromoter']))
        train_vars.append(var_mask(p['Mopromoter'], p['promoter']))
        train_acts.append(p['act'])
    for p in synthetic_promoter[synthetic_percent: ]:
        val_bases.append(onehot_encode(p['Mopromoter']))
        val_vars.append(var_mask(p['Mopromoter'], p['promoter']))
        val_acts.append(p['act'])

    train_set = VarPromoterDataset(train_bases, train_vars, train_acts)
    train_loader = DataLoader(train_set, batch_size = cfg.batch_size, shuffle = True, drop_last = False)
    val_set = VarPromoterDataset(val_bases, val_vars, val_acts)
    val_loader = DataLoader(val_set, batch_size = 1, shuffle = False, drop_last = False)

    # laod test data
    if cfg.have_test:
        test_promoter = []
        with open(cfg.test_data_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for i, [ID, mopromoter, promoter, act] in enumerate(reader):
                if mopromoter in wild_promoter:
                    test_promoter.append({'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))})
                else:
                    test_promoter.append({'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))})
        test_bases, test_vars, test_acts = [], [], []
        for p in test_promoter:
            test_bases.append(onehot_encode(p['Mopromoter']))
            test_vars.append(var_mask(p['Mopromoter'], p['promoter']))
            test_acts.append(p['act'])
        test_set = VarPromoterDataset(test_bases, test_vars, test_acts)
        test_loader = DataLoader(test_set, batch_size = 1, shuffle = False, drop_last = False)
    else:
        test_loader = None

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
        predict_bases, predict_vars = [], []
        for p in predict_promoter:
            predict_bases.append(onehot_encode(p['Mopromoter']))
            predict_vars.append(var_mask(p['Mopromoter'], p['promoter']))
        predict_set = VarPromoterDataset(predict_bases, predict_vars)
        predict_loader = DataLoader(predict_set, batch_size = 1, shuffle = False, drop_last = False)
    else:
        predict_loader = None


    base_promoter = []
    with open(cfg.base_promoter_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, [ID, mopromoter, promoter, act] in enumerate(reader):
            if mopromoter in wild_promoter:
                base_promoter.append({'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))})
            else:
                base_promoter.append({'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))})
    base_bases, base_vars, base_acts = [], [], []
    for p in base_promoter:
        base_bases.append(onehot_encode(p['Mopromoter']))
        base_vars.append(var_mask(p['Mopromoter'], p['promoter']))
        base_acts.append(p['act'])
    base_set = VarPromoterDataset(base_bases, base_vars, base_acts)
    base_loader = DataLoader(base_set, batch_size = 1, shuffle = False, drop_last = False)


    return train_loader, val_loader, test_loader, predict_loader, base_loader

