#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : m6A_extra
@File        : save_class_fea.py
@Author      : Mengya Liu
@Date        : 2024/5/20 20:38
@Description : Ref: SMEP

"""
print(__doc__)

import numpy as np
import pandas as pd
import datetime as dt
from keras.utils import np_utils
from tensorflow.keras.preprocessing import sequence

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

max_seq_len = int("800")
class_num = int("2")


def data_process(in_file, OutputDir, prefix):
    print('Loading data...')

    print(in_file)

    times1 = dt.datetime.now()

    data = pd.read_csv(in_file, index_col=False, header=None)
    print(data.shape)

    y = data[0]  # the same as test. (20220924, comment by LMY)
    x_ori = data[1]
    x = []
    for pi in x_ori:
        nr = pi.split(' ')[0:-1]
        ndata = list(map(int, nr))
        x.append(ndata)
    x = np.array(x)

    times2 = dt.datetime.now()

    print('Time spent: ' + str(times2 - times1))

    # convert class vectors to binary class matrices
    # y_class = np_utils.to_categorical(y, class_num)
    x_class = sequence.pad_sequences(x, maxlen=max_seq_len)

    print(x_class.shape)
    print(y.shape)

    np.save(OutputDir + '/' + prefix + '_class_fea.npy', x_class)
    np.save(OutputDir + '/' + prefix + '_class_y.npy', y)


# Example
for i in range(5):
    data_process('fold'+str(i)+'/train_class_fea', 'fold'+str(i), 'train')
    data_process('fold'+str(i)+'/valid_class_fea', 'fold'+str(i), 'valid')

# data_process('./indep_test_data/maize_test_class_fea', './indep_test_data', 'maize_test')
# data_process('./train_data/train_class_fea', './train_data', 'train')
