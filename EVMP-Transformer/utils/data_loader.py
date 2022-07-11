# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (Transformer)
# Time:     2022.5.18

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
        vars                k-mer of variant subsequence
        indices             index of variations
        acts                promoter fluorescence intensity

    Dataset item:
        with act            ({'base': base, 'var': var, 'index': index}, act)
        no act              {'base': base, 'var': var, 'index': index}
    '''
    def __init__(self, bases, vars, indices, acts = None):
        super(Dataset, self).__init__()
        self.bases = bases
        self.vars = vars
        self.indices = indices
        self.acts = acts

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, idx):
        base = self.bases[idx]
        var = self.vars[idx]
        index = self.indices[idx]
        promoter = {
            'base': np.array(base, dtype = np.float64), 
            'var': np.array(var, dtype = np.float64), 
            'index': np.array(index, dtype = np.float64)
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


def var_subseq(mother_promoter, origin_promoter):
    '''
    variation subsequence (var) and variation position 
    for origin promoter and mother promoter (wild_index th wild promoter).

    if number of variations > `num_var`, then var = None
    if origin promoter = mother promoter, then var = [0, ..., 0], index = -1

    return (vars, indices)
    '''
    vocab = {'B': 0, 'A': 1, 'G': 2, 'C': 3, 'T': 4}
    vocab_size = len(vocab)
    vars, indices = [], []
    origin_promoter += 'B' * (cfg.seq_len - len(origin_promoter))
    mother_promoter += 'B' * (cfg.seq_len - len(mother_promoter))
    for i in range(cfg.seq_len):
        if origin_promoter[i] != mother_promoter[i]:
            mer = [vocab[origin_promoter[i]] if k >= 0 and k < cfg.seq_len else 0 
                            for k in range(i - ((cfg.mer - 1) // 2), i + (cfg.mer - ((cfg.mer - 1) // 2)))]
            var = np.array([one_hot(v, vocab_size) for v in mer]).reshape(-1)
            vars.append(var)
            indices.append(i)
    if len(vars) <= cfg.num_var:
        vars += [np.array(np.tile(one_hot(vocab_size, vocab_size), cfg.mer))] * (cfg.num_var - len(vars))
        indices += [-1] * (cfg.num_var - len(indices))
        return np.array(vars), np.array(indices)
    return None, None


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
            # wild_promoter[promoter] = {'Mopromoter': promoter, 'promoter': promoter, 'act': math.log10(float(act))}
            synthetic_promoter.append({'Mopromoter': mopromoter, 'promoter': promoter, 'act': math.log10(float(act))})

    synthetic_percent = int(len(synthetic_promoter) * (1 - cfg.val_ratio)) // cfg.batch_size * cfg.batch_size
    wilds = list(wild_promoter.values())
    random.shuffle(wilds)
    random.shuffle(synthetic_promoter)
    train_bases, train_vars, train_indices, train_acts = [], [], [], []
    val_bases, val_vars, val_indices, val_acts = [], [], [], []

    for p in wilds:
        train_bases.append(onehot_encode(p['Mopromoter']))
        var, index = var_subseq(p['Mopromoter'], p['promoter'])
        train_vars.append(var)
        train_indices.append(index)
        train_acts.append(p['act'])
    for p in synthetic_promoter[: synthetic_percent]:
        var, index = var_subseq(p['Mopromoter'], p['promoter'])
        if not var is None:
            train_bases.append(onehot_encode(p['Mopromoter']))
            train_vars.append(var)
            train_indices.append(index)
            train_acts.append(p['act'])
    for p in synthetic_promoter[synthetic_percent: ]:
        var, index = var_subseq(p['Mopromoter'], p['promoter'])
        if not var is None:
            val_bases.append(onehot_encode(p['Mopromoter']))
            val_vars.append(var)
            val_indices.append(index)
            val_acts.append(p['act'])

    train_set = VarPromoterDataset(train_bases, train_vars, train_indices, train_acts)
    train_loader = DataLoader(train_set, batch_size = cfg.batch_size, shuffle = True, drop_last = False)
    val_set = VarPromoterDataset(val_bases, val_vars, val_indices, val_acts)
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
        test_bases, test_vars, test_indices, test_acts = [], [], [], []
        for p in test_promoter:
            var, index = var_subseq(p['Mopromoter'], p['promoter'])
            if not var is None:
                test_bases.append(onehot_encode(p['Mopromoter']))
                test_vars.append(var)
                test_indices.append(index)
                test_acts.append(p['act'])
        test_set = VarPromoterDataset(test_bases, test_vars, test_indices, test_acts)
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
        predict_bases, predict_vars, predict_indices = [], [], []
        for p in predict_promoter:
            var, index = var_subseq(p['Mopromoter'], p['promoter'])
            if not var is None:
                predict_bases.append(onehot_encode(p['Mopromoter']))
                predict_vars.append(var)
                predict_indices.append(index)
        predict_set = VarPromoterDataset(predict_bases, predict_vars, predict_indices)
        predict_loader = DataLoader(predict_set, batch_size = 1, shuffle = False, drop_last = False)
    else:
        predict_loader = None

    return train_loader, val_loader, test_loader, predict_loader

