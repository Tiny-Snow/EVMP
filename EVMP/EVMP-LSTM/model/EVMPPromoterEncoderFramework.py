# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (LSTM)
# Time:     2022.5.7


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import config as cfg
from utils.helper import Signal, log
from utils.train import *
from utils.plot import *

INF = float('inf')


class EVMPPromoterEncoderFramework(nn.Module):
    '''
        Extended Vision Mutation Priority Promoter Encoder Framework (LSTM)

        device              device of model
        dropout             dropout ratio of linear layer
        LSTM_dropout        dropout ratio of LSTM layer
        base_embed_size     base promoter embedding size
        var_embed_size      variation embedding size
        var_units           units of var LSTM layer
        var_layers          number of var LSTM layers
    
    '''
    def __init__(self, device, dropout, LSTM_dropout, base_embed_size = 1000, base_layers = 5, var_embed_size = 200, var_units = 200, var_layers = 5):
        super(EVMPPromoterEncoderFramework, self).__init__()

        # get device 
        self.device = device
        log('>>>>> DEVICE: {}'.format(self.device))

        # signal
        self.sig = Signal(cfg.signal_file)

        # base embedding
        self.base_layers = base_layers
        self.base_embed = nn.LSTM(5, base_embed_size, num_layers = base_layers, bidirectional = False, dropout = LSTM_dropout, batch_first = True)
        # variation embedding
        self.var_embed = nn.Embedding(cfg.max_wild * (cfg.seq_len * (5 ** cfg.mer) + 1), var_embed_size)
        # variation LSTM representation
        self.var_units = var_units
        self.var_layers = var_layers
        self.varLSTM = nn.LSTM(var_embed_size, var_units, num_layers = var_layers, bidirectional = False, dropout = LSTM_dropout, batch_first = True)
        # FC layers
        self.fcs = [nn.Linear(var_units, var_embed_size)] * var_layers
        self.actives = [nn.ReLU()] * var_layers
        self.dropouts = [nn.Dropout(p = dropout)] * var_layers
        # base + var representation FC output
        self.output = nn.Linear(base_embed_size * base_layers + var_embed_size * var_layers, 1)


    def forward(self, x):
        ''' 
        Given input x, compute output of the network 
            
        intput x: 
        x.base      base promoter one-hot encoding
        x.var       variation index sequence
        '''
        base, var = x['base'].float().to(self.device), x['var'].float().to(self.device)
        # base: [bacth_size, seq_len, 5] ->  [bacth_size, base_embed_size]   
        o, (h, base_embedding) = self.base_embed(base)
        base_embedding = base_embedding.view([base_embedding.size()[1], -1])
        # var: [bacth_size, seq_len] (index of var) -> [bacth_size, num_var, var_embed_size]
        var = self.var_embed(var.long())
        _, (hn, cn) = self.varLSTM(var)
        var_outs = []
        for layer in range(self.var_layers):
            self.fcs[layer].cuda()
            var_out = self.fcs[layer](cn[layer])
            var_out = self.actives[layer](var_out)
            var_out = self.dropouts[layer](var_out)
            var_outs.append(var_out)
        var_rep = torch.cat(var_outs, dim = -1)
        represent = torch.cat([var_rep, base_embedding], dim = -1)
        out = self.output(represent.to(self.device))
        out = out.squeeze(-1)
        return out


def run(train_loader, val_loader, test_loader, predict_loader = None): 
    '''
    Train the model and test the data
    '''
    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EVMPPromoterEncoderFramework(device = device, dropout = cfg.dropout, LSTM_dropout = cfg.LSTM_dropout).to(device)
    # signal
    signal = Signal(cfg.signal_file)
    # criterion
    criterion = nn.L1Loss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr, weight_decay = cfg.weight_decay)
    # lr scheduler
    # scheduler = StepLR(optimizer, cfg.LRstep_size, cfg.LRgamma)
    now_epoch = cfg.LRstep_size
    epoch_step = cfg.LRstep_size
    milestones = []
    while now_epoch <= cfg.num_epoch:
        milestones.append(now_epoch)
        epoch_step = int(epoch_step * cfg.LRstep_alpha)
        now_epoch += epoch_step
    scheduler = MultiStepLR(optimizer, milestones = milestones, gamma = cfg.LRgamma)

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
    result = output(y_train, pred_train, y_valid, pred_valid, y_test, pred_test, file_name = 'EVMP-LSTM')

    log("Final Model: ")
    log("    train MAE: {:3.6f}, train R2: {:.2f}".format(result['train'][0], result['train'][1]))
    log("    valid MAE: {:3.6f}, valid R2: {:.2f}".format(result['valid'][0], result['valid'][1]))
    log("    test  MAE: {:3.6f}, test  R2: {:.2f}".format(result['test'][0], result['test'][1]))

    # predict
    if cfg.if_predict:
        log('>>>>> Testing Complete! Start Predicting...')
        pred = predict(device, model, predict_loader)
        save_predict(pred)
