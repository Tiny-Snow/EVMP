# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (LSTM)
# Time:     2022.5.7

import os
import config as cfg
import random
import numpy as np
import torch


class Signal:
    '''Running signal to control training process'''

    def __init__(self, signal_file):
        self.signal_file = signal_file
        self.train = True
        self.update()

    def update(self):
        signal_dict = self.read_signal()
        self.train = signal_dict['train']

    def read_signal(self):
        with open(self.signal_file, 'r') as fin:
            return eval(fin.read())


def log(message):
    '''Log --> log file'''
    if os.path.exists(cfg.log_file):
        with open(cfg.log_file, 'a') as f:
            f.write(message + '\n')
    else:
        with open(cfg.log_file, 'w') as f:
            f.write(message + '\n')
    print(message)


def create_dir(path):
    '''Create directory if not exists'''
    if not os.path.exists(path):
        os.makedirs(path)


def seed_all(seed = 2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
