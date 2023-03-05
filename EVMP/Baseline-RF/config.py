# -*- coding:UTF-8 -*-
# Author:   Raffica
# Project:  EVMP BASELINES -- RF
# Time:     2022.7.9

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


#------------------------ info ------------------------#
info = '2023'
seed = 2023
fold = 5

#------------------------ data ------------------------#
data_root = BASE_DIR + '/data/'
wild_data_path = data_root + 'wild_promoter.csv'                # wild promoter data
synthetic_data_path = data_root + 'synthetic_promoter.csv'      # synthetic promoter data
val_ratio = 0.10        # validation set ratio (for synthetic promoter)
test_ratio = 0.10       # test set ratio (for synthetic promoter)
seq_len = 85            # max base promoter size

#------------------------ train ------------------------#

#------------------------ save ------------------------#
save_root = BASE_DIR + '/save'                              # save model and log
log_file = save_root + '/log_{}.txt'.format(info)           # save log file path
save_fig = BASE_DIR + '/output/RF_{}.pdf'.format(info)      # save train/valid/test figure path
