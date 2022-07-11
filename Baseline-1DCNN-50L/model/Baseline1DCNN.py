# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- 1D CNN (Layer 50)
# Time:     2022.7.9

import os
import math

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import optimize
import matplotlib.pyplot as plt
import csv

import config as cfg
from tqdm import tqdm
from utils.helper import Signal, log

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
    def __init__(self, device, seq_len, input_size, hidden_size = [1024, 128, 1024, 128]):
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
            x = self.relu(self.cnn1d[i](x))
        x = x.view(batch_size, -1).to(self.device)
        x = self.fc(x)
        return x.reshape(-1)


def train(device, model, train_loader, criterion, optimizer):
    '''Train model, return train loss'''
    train_loss = 0.0
    model.train()       # set the model to training mode
    bar = tqdm(enumerate(train_loader), desc = 'Train', total = len(train_loader))
    for i, data in bar:
        inputs, values = data
        values = values.float().to(device)
        optimizer.zero_grad() 
        outputs = model(inputs) 
        batch_loss = criterion(outputs, values)
        batch_loss.backward() 
        optimizer.step() 
        train_loss += batch_loss.item()
        bar.set_description('Train Loss: {:.4f}'.format(train_loss / (i + 1)))
    return train_loss


def valid(device, model, val_loader, criterion, optimizer):
    '''Validate model, return val loss'''
    val_loss = 0.0
    model.eval()        # set the model to evaluation mode
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, values = data
            values = values.float().to(device)
            outputs = model(inputs) 
            batch_loss = criterion(outputs, values) 
            val_loss += batch_loss.item()
    return val_loss


def _test(device, model, test_loader):
    '''Test model, return true value and pred value'''
    y_true, y_pred = [], []
    model.eval()        # set the model to evaluation mode
    with torch.no_grad():
        for data, value in test_loader:
            inputs = data
            outputs = model(inputs) 
            for y in value.cpu().numpy():
                y_true.append(y)
            for y in outputs.cpu().numpy():
                y_pred.append(y)
    return y_true, y_pred


def test(device, model, test_loader, criterion):
    '''Test model, save MAE and R2'''
    test_predict = []
    test_predict_norm = []
    model.eval()        # set the model to evaluation mode
    with torch.no_grad():
        for data, value in test_loader:
            inputs = data
            outputs = model(inputs) 
            for y in outputs.cpu().numpy():
                test_predict.append([math.pow(10, y), math.pow(10, value)])
                test_predict_norm.append([y, value])
    test_predict = torch.Tensor(np.array(test_predict))
    test_predict_norm = torch.Tensor(np.array(test_predict_norm, dtype = np.float32))
    # loss
    loss = criterion(test_predict[:, 0], test_predict[:, 1]).item()
    loss_norm = criterion(test_predict_norm[:, 0], test_predict_norm[:, 1]).item()
    # R2
    r2 = r2_score(test_predict[:, 0], test_predict[:, 1])
    r2_norm = r2_score(test_predict_norm[:, 0], test_predict_norm[:, 1])
    log(">>>>> Test MAE: {}".format(loss))
    log(">>>>> Test MAE_norm: {}".format(loss_norm))
    log(">>>>> Test R2 Score: {}".format(r2))
    log(">>>>> Test R2_norm Score: {}".format(r2_norm))
    # save result
    if os.path.exists(cfg.test_result_path):
        os.remove(cfg.test_result_path)
    with open(cfg.test_result_path, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'ACT_Predict', 'ACT_True'])
        for i, p in enumerate(test_predict_norm):
            y_predict, y_true = p
            y_predict, y_true = np.round(y_predict, 4).numpy(), np.round(y_true, 4).numpy()
            writer.writerow([i, y_predict, y_true]) 
        writer.writerow(['MAE', loss])
        writer.writerow(['MAE_norm', loss_norm])
        writer.writerow(['R2 Score', r2])
        writer.writerow(['R2_norm Score', r2_norm])
    # plot
    plt.title('R2=' + str(round(r2_norm, 2)) + ', MAE=' + str(round(loss_norm, 2)))
    plt.scatter(test_predict_norm[:, 1], test_predict_norm[:, 0], c="pink", marker='o')
    plt.xlabel('Origin')
    plt.ylabel('Predict')
    plt.savefig(cfg.save_fig)


def predict(device, model, predict_loader):
    predicts = []
    model.eval()        # set the model to evaluation mode
    with torch.no_grad():
        for data in predict_loader:
            inputs = data
            outputs = model(inputs) 
            for y in outputs.cpu().numpy():
                predicts.append(math.pow(10, y))
    # save result
    if os.path.exists(cfg.predict_result_path):
        os.remove(cfg.predict_result_path)
    with open(cfg.predict_result_path, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Act_Predict'])
        for i, p in enumerate(predicts):
            y_predict = p
            y_predict = np.round(y_predict, 4)
            writer.writerow([i, y_predict]) 


def line(x, A, B):
    return A * x + B

def output(y_train, pred_train, y_test, pred_test, file_name = '1DCNN (50 layers)'):
    '''Output the prediction result of the train / valid'''
    score_train_MAE = mean_absolute_error(y_train, pred_train)
    score_train_r2 = r2_score(y_train, pred_train)
    score_test_MAE = mean_absolute_error(y_test, pred_test)
    score_test_r2 = r2_score(y_test, pred_test)

    log("Final Model: ")
    log("    train MAE: {:3.6f}, train R2: {:.2f}".format(score_train_MAE, score_train_r2))
    log("    valid MAE: {:3.6f}, valid R2: {:.2f}".format(score_test_MAE, score_test_r2))
    
    x = np.arange(min(y_train), max(y_train), 0.01)
    A1, B1 = optimize.curve_fit(line, y_test, pred_test)[0]
    y1 = A1 * x + B1
    
    plt.figure()
    plt.grid()
    plt.plot(x, y1, c = "red", linewidth = 3.0)
    plt.title(file_name)
    plt.scatter(y_train, pred_train, s = 3, c = "silver", marker = 'o', 
        label = "Train: $MAE$ = {:.2f}, $R^2$ = {:.2f}".format(score_train_MAE, score_train_r2))
    plt.scatter(y_test, pred_test, s = 5, c = "darkgreen", marker = 'o', 
        label = "Valid: $MAE$ = {:.2f}, $R^2$ = {:.2f}".format(score_test_MAE, score_test_r2))
    plt.xlim(1, 5)
    plt.ylim(1, 5)
    plt.xlabel('True Value')
    plt.ylabel('Predict Value')
    plt.legend()
    plt.savefig(cfg.save_train)


def run(train_loader, val_loader, test_loader, predict_loader = None):
    '''
    Train the model and test the data
    '''
    # model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = [1024, 128] * 25
    model = Baseline1DCNN(device = device, seq_len = cfg.seq_len, input_size = 5, hidden_size = hidden_size).to(device)
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
                    val_loss = valid(device, model, val_loader, criterion, optimizer)
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

    # output
    y_train, pred_train = _test(device, model, train_loader)
    y_test, pred_test = _test(device, model, val_loader)
    output(y_train, pred_train, y_test, pred_test)

    log('>>>>> Training Complete! Start Testing...')

    # create model and load weights from best checkpoint
    if not cfg.if_test:
        model.load_state_dict(torch.load(cfg.model_path))
    else:
        model.load_state_dict(torch.load(cfg.pretrain_model_path))

    # test
    if cfg.have_test:
        test(device, model, test_loader, criterion)

    # predict
    if cfg.if_predict:
        log('>>>>> Testing Complete! Start Predicting...')
        predict(device, model, predict_loader)