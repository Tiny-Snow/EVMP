# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (XGBoost)
# Time:     2022.7.11

import math
import random
import numpy as np
import pandas as pd
import torch
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


def pack_base_and_var(bases, vars, acts):
    '''
    pack base promoter one-hot encoding and masked var promoter in a np.array
    return each synthetic promoter: 
        {
            input: np.array([base promoter one-hot (5 * `seq_len`)] + [var promoter one-hot (5 * `seq_len`)])
            value: act (int)
        }
    '''
    packs_input, packs_value = [], []
    for base, var, act in zip(bases, vars, acts):
        base = base.flatten()
        var = var.flatten()
        packs_input.append(np.concatenate((base, var)))
        packs_value.append(act)
    return packs_input, packs_value


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

    train_bases, train_vars, train_acts  = [], [], []
    test_bases, test_vars, test_acts      = [], [], []

    # [test, train, valid]
    test_percent = int(len(synthetic_promoter) * cfg.test_ratio)

    for p in wilds:
        train_bases.append(onehot_encode(p['Mopromoter']))
        train_vars.append(var_mask(p['Mopromoter'], p['promoter']))
        train_acts.append(p['act'])
    for p in synthetic_promoter[: test_percent]:
        test_bases.append(onehot_encode(p['Mopromoter']))
        test_vars.append(var_mask(p['Mopromoter'], p['promoter']))
        test_acts.append(p['act'])
    for p in synthetic_promoter[test_percent: ]:
        train_bases.append(onehot_encode(p['Mopromoter']))
        train_vars.append(var_mask(p['Mopromoter'], p['promoter']))
        train_acts.append(p['act'])

    train_input, train_value = pack_base_and_var(train_bases, train_vars, train_acts)
    test_input, test_value   = pack_base_and_var(test_bases, test_vars, test_acts)

    return train_input, train_value, test_input, test_value

    # TODO: load predict data
