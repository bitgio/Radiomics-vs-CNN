# -*- coding: utf-8 -*-
#
# Copyright (C) 2023 Giovanni Bitonti & Ana Pascual
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

Codice Principale per la costruzione della CNN
-----------------------------------------------
"""


import os
import threading as thr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, InputLayer, Activation, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD
from .modulo_0 import matlab_on
from .modulo_1 import create_dataset


if __name__ == '__main__':

    mammo_o, mammo_f, label = [], [], []
    data_folder = "./dataset/"
    os.chdir(data_folder)
    listdir = os.listdir()

    os.chdir("../")
    threads = []
    chunk = 6

    for i in range(49):
        t = thr.Thread(target = create_dataset, args = (listdir[i*chunk : (i+1)*chunk], mammo_o, mammo_f, label))
        threads.append(t)
        t.start()
    
    for j in threads:
        j.join()

    mammo_o = np.asarray(mammo_o, dtype = 'float32')/255.
    mammo_f = np.asarray(mammo_f, dtype = 'float32')/255.
    label = np.asarray(label)

    mammo_o_4d = np.reshape(mammo_o, (147, 125, 125, 1))
    mammo_f_4d = np.reshape(mammo_f, (147, 64, 64, 1))



