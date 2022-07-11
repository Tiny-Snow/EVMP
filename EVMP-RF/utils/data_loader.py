# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (RF)
# Time:     2022.7.11

import pandas as pd
import numpy as np
import torch

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


def pack_base_and_var(bases, vars, indices, acts):
    '''
    pack base promoter one-hot encoding and var indices + subseq in a np.array
    return each synthetic promoter: 
        {
            'base': np.array([base promoter one-hot (5 * `seq_len`)] + [var index (int), var subseq one-hot (5 * mer)])
            'act': act (int)
        }
    '''
    packs_input, packs_value = [], []
    for base, var, index, act in zip(bases, vars, indices, acts):
        base = base.flatten()
        var = var.flatten()
        packs_input.append(np.concatenate((base, var, index)))
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
    bases, vars, indices, acts = [], [], [], []

    for p in wilds:
        bases.append(onehot_encode(p['Mopromoter']))
        var, index = var_subseq(p['Mopromoter'], p['promoter'])
        vars.append(var)
        indices.append(index)
        acts.append(p['act'])
    for p in synthetic_promoter:
        var, index = var_subseq(p['Mopromoter'], p['promoter'])
        if not var is None:
            bases.append(onehot_encode(p['Mopromoter']))
            vars.append(var)
            indices.append(index)
            acts.append(p['act'])

    return pack_base_and_var(bases, vars, indices, acts)

