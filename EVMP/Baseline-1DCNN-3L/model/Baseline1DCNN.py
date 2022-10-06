# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- 1D CNN (3 layers)
# Time:     2022.7.9

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import config as cfg
from utils.helper import Signal, log
from utils.train import *
from utils.plot import *

INF = float('inf')


class Baseline1DCNN(nn.Module):
    '''
    EVMP Baseline: 1D CNN
    Args:
        device              device of model
        seq_len             length of sequence
        input_size          input channels
        hidden_size         hidden channels of each CNN layer (list)
    '''
    def __init__(self, device, seq_len, input_size, hidden_size = [1024, 512, 128]):
        super().__init__()
        self.device = device
        self.cnn1d = nn.ModuleList()
        self.relu = nn.ReLU()
        hidden_size = [input_size] + hidden_size
        for i in range(len(hidden_size) - 1):
            self.cnn1d.append(nn.Conv1d(in_channels = hidden_size[i], out_channels = hidden_size[i + 1], kernel_size = 3, padding = 1))
        self.fc = nn.Linear(in_features = hidden_size[-1] * seq_len, out_features = 1)


    def forward(self, x):
        ''' 
        Given input x, compute output of the network 
        '''
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1).float().to(self.device)
        for i in range(len(self.cnn1d)):
            x = self.cnn1d[i](x)
            x = self.relu(x)
        x = x.view(batch_size, -1).to(self.device)
        x = self.fc(x)
        return x.reshape(-1)


def run(train_loader, val_loader, test_loader, predict_loader = None):
    '''
    Train the model and test the data
    '''
    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Baseline1DCNN(device = device, seq_len = cfg.seq_len, input_size = 5, hidden_size = [1024, 512, 128]).to(device)
    # signal
    signal = Signal(cfg.signal_file)
    # criterion
    criterion = nn.L1Loss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr, weight_decay = cfg.weight_decay)
    # lr scheduler
    # scheduler = StepLR(optimizer, cfg.LRstep_size, cfg.LRgamma)
    now_epoch = cfg.LRstep_size
    eopch_step = cfg.LRstep_size
    milestones = []
    while now_epoch <= cfg.num_epoch:
        milestones.append(now_epoch)
        eopch_step = int(eopch_step * cfg.LRstep_alpha)
        now_epoch += eopch_step
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
    result = output(y_train, pred_train, y_valid, pred_valid, y_test, pred_test, file_name = '1DCNN (3 layers)')

    log("Final Model: ")
    log("    train MAE: {:3.6f}, train R2: {:.2f}".format(result['train'][0], result['train'][1]))
    log("    valid MAE: {:3.6f}, valid R2: {:.2f}".format(result['valid'][0], result['valid'][1]))
    log("    test  MAE: {:3.6f}, test  R2: {:.2f}".format(result['test'][0], result['test'][1]))

    # predict
    if cfg.if_predict:
        log('>>>>> Testing Complete! Start Predicting...')
        pred = predict(device, model, predict_loader)
        save_predict(pred)