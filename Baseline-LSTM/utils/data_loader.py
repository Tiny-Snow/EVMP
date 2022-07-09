# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  EVMP BASELINES -- LSTM
# Time:     2022.7.9

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
        indicex:            index of variations
        acts                promoter fluorescence intensity

    Dataset item:
        with act:           ({'base': base, 'var': var, 'index': index}, act)
        no act:             {'base': base, 'var': var, 'index': index}
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
        if i < len(promoter):
            onehots.append(one_hot(vocab[promoter[i]], vocab_size))
        else:
            onehots.append(one_hot(vocab_size, vocab_size))
    return np.array(onehots, dtype = np.float64)


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
                            for k in range(i - ((cfg.k_mer - 1) // 2), i + (cfg.k_mer - ((cfg.k_mer - 1) // 2)))]
            var = np.array([one_hot(v, vocab_size) for v in mer]).reshape(-1)
            vars.append(var)
            indices.append(i)
    if len(vars) <= cfg.num_var:
        vars += [np.array(np.tile(one_hot(vocab_size, vocab_size), cfg.k_mer))] * (cfg.num_var - len(vars))
        indices += [-1] * (cfg.num_var - len(indices))
        return np.array(vars), np.array(indices)
    return None, None


def load_data():
    '''
    Load train/val/test/predict data
    '''
    # load train data and test data
    synthetic_data, test_data = [], []
    for dataset_pair in [(cfg.synthetic_data_path, synthetic_data), (cfg.test_data_path, test_data)]:
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
    test_loader = DataLoader(test_data, batch_size = 1, shuffle = False)

    return train_loader, val_loader, test_loader, None
