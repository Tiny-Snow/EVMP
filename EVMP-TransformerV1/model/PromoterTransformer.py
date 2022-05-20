# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (Transformer Version 1)
# Time:     2022.5.18


import torch
import torch.nn as nn
import math


class BasePositionalEncoding(nn.Module):
    '''
    Base Promoter Positional Encoding for BaseTransformer
    
    Args:
        d_model         base nucleotide embedding size
        seq_len         max length of promoter
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


class VarPositionalEncoding(nn.Module):
    '''
    Variation Positional Encoding for VarTransformer
    
    Args:
        d_model         var embedding size
        num_var         length of var sequence
        seq_len         max length of promoter
    '''
    def __init__(self, d_model, num_var, seq_len, dropout = 0.1):
        super(VarPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        self.seq_len = seq_len
        self.num_var = num_var

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term[: d_model // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer('pe', pe)

    def forward(self, x, index):
        ''' 
        input:
            x: variation ([batch_size, num_var, d_model])
            index: position of variation ([batch_size, num_var])
            if index = -1, that means no variation, no need for positional encoding 
        '''
        for batch in range(x.size(0)):
            for k in range(x.size(1)):
                if index[batch][k] != -1:
                    x[batch, k] += self.pe[index[batch][k].long()]
        return self.dropout(x)

