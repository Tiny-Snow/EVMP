# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- RF
# Time:     2022.7.9

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


#------------------------ info ------------------------#
info = ''

#------------------------ data ------------------------#
data_root = BASE_DIR + '/data/'
wild_data_path = data_root + 'wild_promoter.csv'                # wild promoter data
synthetic_data_path = data_root + 'synthetic_promoter.csv'      # synthetic promoter data
val_ratio = 0.10        # validation set ratio (for synthetic promoter)
seq_len = 85            # max base promoter size

#------------------------ train ------------------------#
cross_time = 10         # cross validation times
param_xgb = {           # params for XGBoost
    'n_estimators': 500,        # number of trees
    'learning_rate': 0.1, 
    'max_depth': 5,             # max depth of tree
    'min_child_weight': 1,      # min weight of leaf
    'seed': 1000,
    'subsample': 0.8,           # dataset subsample
    'colsample_bytree': 0.7,    # column sample
    'reg_lambda': 1,            # regularization coefficient
    'silent': 1                 
}
param_grid = {          # params for GridSearchCV
    'n_estimators': [10, 50, 100, 300], 
    'learning_rate': [0.2, 0.1, 0.05],
    'max_depth': [3, 5, 7]
}

#------------------------ save ------------------------#
save_root = BASE_DIR + '/save'                              # save model and log
log_file = save_root + '/log_{}.txt'.format(info)           # save log file path
fig_path = BASE_DIR + '/output'                             # save figure path
