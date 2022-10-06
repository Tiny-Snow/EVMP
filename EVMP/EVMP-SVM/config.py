# -*- coding:UTF-8 -*-
# Author:   Tiny Snow
# Project:  Extended Vision Mutation Priority Promoter Encoder Framework (SVM)
# Time:     2022.7.11

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


#------------------------ info ------------------------#
info = ''

#------------------------ data ------------------------#
data_root = BASE_DIR + '/data/'
wild_data_path = data_root + 'wild_promoter.csv'                # wild promoter data
synthetic_data_path = data_root + 'synthetic_promoter.csv'      # synthetic promoter data
val_ratio = 0.10        # validation set ratio (for synthetic promoter)
test_ratio = 0.10       # test set ratio (for synthetic promoter)
seq_len = 85            # max base promoter size
num_var = 10            # max variation size
mer = 8                 # k-mer for Extended Vision

#------------------------ train ------------------------#
cross_time = 10         # cross validation times

#------------------------ save ------------------------#
save_root = BASE_DIR + '/save'                              # save model and log
log_file = save_root + '/log_{}.txt'.format(info)           # save log file path
fig_path = BASE_DIR + '/output'                             # save figure path
