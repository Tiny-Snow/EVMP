# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (Transformer)
# Time:     2022.5.18

import os

import torch
from tqdm import tqdm
import csv

import config as cfg


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


def valid(device, model, val_loader, criterion):
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


def test(device, model, test_loader):
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


def predict(device, model, predict_loader):
    '''predict model, return predict value'''
    predicts = []
    model.eval()        # set the model to evaluation mode
    with torch.no_grad():
        for data in predict_loader:
            inputs = data
            outputs = model(inputs) 
            for y in outputs.cpu().numpy():
                predicts.append(y)
    return predicts


def save_predict(predicts, idx = 0):
    '''write predict value to csv'''
    if os.path.exists(cfg.predict_result_path.replace('.csv', '_C{}.csv'.format(idx))):
        os.remove(cfg.predict_result_path.replace('.csv', '_C{}.csv'.format(idx)))
    with open(cfg.predict_result_path.replace('.csv', '_C{}.csv'.format(idx)), 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Pred'])
        for i, p in enumerate(predicts):
            writer.writerow([i, p])
