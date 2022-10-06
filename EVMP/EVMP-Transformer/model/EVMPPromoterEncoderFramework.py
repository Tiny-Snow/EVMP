# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (Transformer)
# Time:     2022.5.18

import torch
import torch.nn as nn
from .PromoterTransformer import BasePositionalEncoding, VarPositionalEncoding
from utils.CycleLR import CycleDecayLR

import config as cfg
from utils.helper import Signal, log
from utils.train import *
from utils.plot import *

INF = float('inf')


class EVMPPromoterEncoderFramework(nn.Module):
    '''
    Extended Vision Mutation Priority Promoter Encoder Framework (Transformer)

    Args:
        device              device of model
        mer                 k-mer of extended vision variation
        seq_len             max length of base promoter
        num_var             max number of variations
        base_layers         number of base promoter encoder layers
        base_embed_size     embedding size of one nucleotide in base promoter
        var_layers          number of variation encoder layers
        var_embed_size      embedding size of each mer in each variation
    
    '''
    def __init__(self, device, mer, seq_len, num_var, base_layers = 6, base_embed_size = 32, var_layers = 6, var_embed_size = 8):
        super(EVMPPromoterEncoderFramework, self).__init__()
        # get device
        self.device = device
        log('>>>>> DEVICE: {}'.format(self.device))
        
        # vocab
        self.vocab = {'B': 0, 'A': 1, 'G': 2, 'C': 3, 'T': 4}
        self.vocab_size = len(self.vocab)
        self.seq_len = seq_len
        self.mer = mer
        self.num_var = num_var

        # base embedding
        self.base_embed_size = base_embed_size
        self.base_layers = base_layers
        self.base_pos = BasePositionalEncoding(d_model = self.vocab_size, seq_len = seq_len)
        self.base_encoder_layer = nn.TransformerEncoderLayer(d_model = self.base_embed_size, nhead = 8, dim_feedforward = 2048, batch_first = True) 
        self.base_embed = nn.TransformerEncoder(self.base_encoder_layer, base_layers)

        # variation embedding
        self.var_embed_size = var_embed_size * mer
        self.var_layers = var_layers
        self.var_pos = VarPositionalEncoding(d_model = self.vocab_size * mer, num_var = num_var, seq_len = seq_len)
        self.var_encoder_layer = nn.TransformerEncoderLayer(d_model = self.var_embed_size, nhead = 8, dim_feedforward = 2048, batch_first = True) 
        self.var_embed = nn.TransformerEncoder(self.var_encoder_layer, var_layers)
        
        # base + var representation FC
        self.rep_size = self.base_embed_size * seq_len + self.var_embed_size * num_var
        # output
        self.output = nn.Linear(self.rep_size, 1)

    def forward(self, x):
        ''' 
        Given input x, compute output of the network 
            
        intput x: 
            x.base      base promoter one-hot encoding
            x.var       variation k-mer sequence
            x.index     position of variation
        '''
        base, var, index = \
            x['base'].float().to(self.device), x['var'].float().to(self.device), x['index'].float().to(self.device)
        
        # base: [bacth_size, seq_len, 5] ->  [bacth_size, seq_len, base_embed_size]  
        base = self.base_pos(base)
        base = torch.cat([base, torch.zeros(list(base.size()[: - 1]) + [self.base_embed_size - base.size(-1)]).to(self.device)], dim = -1)
        base_embedding = self.base_embed(base)
        base_embedding = base_embedding.view([base_embedding.size(0), -1])
        
        # var: [bacth_size, num_var, mer * 5] -> [bacth_size, num_var, var_embed_size]
        var = self.var_pos(var, index)
        var = torch.cat([var, torch.zeros(list(var.size()[: -1]) + [self.var_embed_size - var.size(-1)]).to(self.device)], dim = -1)
        var_embedding = self.var_embed(var)
        var_embedding = var_embedding.view([var_embedding.size(0), -1])

        # represent and output
        represent = torch.cat([base_embedding, var_embedding], dim = -1).to(self.device)
        out = self.output(represent)
        out = out.squeeze(-1)
        return out


def run(train_loader, val_loader, test_loader, predict_loader = None):
    '''
    Train the model and test the data
    '''
    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EVMPPromoterEncoderFramework(device = device, mer = cfg.mer, seq_len = cfg.seq_len, num_var = cfg.num_var).to(device)
    model.device = device
    # signal
    signal = Signal(cfg.signal_file)
    # criterion
    criterion = nn.L1Loss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr, weight_decay = cfg.weight_decay)
    # lr scheduler
    scheduler = CycleDecayLR(optimizer, step_size = cfg.LRstep_size, cycle_size = cfg.LRcycle_size, gamma = cfg.LRgamma)

    if cfg.pretrain:
        log('>>>>> Load pre-trained model: {}'.format(cfg.pretrain_model_path))
        model.load_state_dict(torch.load(cfg.pretrain_model_path, map_location = '{}'.format(model.device)))

    log('>>>>> Start  Training...')
    best_loss = INF
    if not cfg.if_test:
        for epoch in range(cfg.num_epoch):
            signal.update()
            if signal.train:
                # training
                train_loss = train(device, model, train_loader, criterion, optimizer)
                # validation
                if len(val_loader) > 0:
                    val_loss = valid(device, model, val_loader, criterion)
                    log('[{:03d}/{:03d}] Train MAE Loss: {:3.6f} | Val MAE loss: {:3.6f}'.format(
                        epoch + 1, cfg.num_epoch, train_loss/len(train_loader), val_loss/len(val_loader)
                    ))
                    # if the model improves, save a checkpoint at this epoch
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(model.state_dict(), cfg.model_path)
                        log('>>>>> Saving model with loss {:.6f}'.format(best_loss/len(val_loader)))
                else:
                    log('[{:03d}/{:03d}] Train Loss: {:3.6f}'.format(
                        epoch + 1, cfg.num_epoch, train_loss/len(train_loader)
                    ))
                scheduler.step()
                log('Learning rate -> {:e}'.format(scheduler.get_last_lr()[0]))
            else:
                log('>>>>> End with signal!')
                break
        # if not validating, save the last epoch
        if len(val_loader) == 0:
            torch.save(model.state_dict(), cfg.model_path)
            log('>>>>> Saving model at last epoch')

    log('>>>>> Training Complete! Start Testing...')

    # create model and load weights from best checkpoint
    if not cfg.if_test:
        model.load_state_dict(torch.load(cfg.model_path))
    else:
        model.load_state_dict(torch.load(cfg.pretrain_model_path))

    # test and output
    y_train, pred_train = test(device, model, train_loader)
    y_valid, pred_valid = test(device, model, val_loader)
    y_test, pred_test = test(device, model, test_loader)
    result = output(y_train, pred_train, y_valid, pred_valid, y_test, pred_test, file_name = 'EVMP-Transformer')

    log("Final Model: ")
    log("    train MAE: {:3.6f}, train R2: {:.2f}".format(result['train'][0], result['train'][1]))
    log("    valid MAE: {:3.6f}, valid R2: {:.2f}".format(result['valid'][0], result['valid'][1]))
    log("    test  MAE: {:3.6f}, test  R2: {:.2f}".format(result['test'][0], result['test'][1]))

    # predict
    if cfg.if_predict:
        log('>>>>> Testing Complete! Start Predicting...')
        pred = predict(device, model, predict_loader)
        save_predict(pred)
