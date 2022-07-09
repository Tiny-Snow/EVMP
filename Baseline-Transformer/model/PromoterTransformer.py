# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  EVMP BASELINES -- Transformer
# Time:     2022.7.9

import torch
import torch.nn as nn
import math


class BasePositionalEncoding(nn.Module):
    '''
    Base Promoter Positional Encoding for BaseTransformer
    
    Args:
        d_model: base nucleotide embedding size
        seq_len: max length of promoter
    '''
    def __init__(self, d_model, seq_len, dropout = 0.1):
        super(BasePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        self.seq_len = seq_len

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term[: d_model // 2 + 1])
        pe[0, :, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        ''' 
        input:
            x: base promoter ([batch_size, seq_len, d_model])
        '''
        x = x + self.pe[0, :x.size(1)]
        return self.dropout(x)
