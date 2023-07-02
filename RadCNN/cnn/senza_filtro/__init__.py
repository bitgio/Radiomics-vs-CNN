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

-----------------------------------------------
Codice principale per la costruzione della CNN
-----------------------------------------------

In questo Python file si può trovare l'implementazione della CNN


"""


import os
import threading as thr
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, InputLayer, Activation, Dropout
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD


__all__ = ['create_nf_dataset', 'nf_cnn']

from modulo_3 import create_nf_dataset
from modulo_4 import nf_cnn


if __name__ == '__main__':

    mammo_o, mammo_f, label = [], [], []
    data_folder = "./dataset/"
    os.chdir(data_folder)
    listdir = os.listdir()

    os.chdir("../")
    threads = []
    chunk = 6

    for i in range(49):
        t = thr.Thread(target = create_nf_dataset, args = (listdir[i*chunk : (i+1)*chunk], mammo_o, mammo_f, label))
        threads.append(t)
        t.start()
    
    for j in threads:
        j.join()

    mammo_o = np.asarray(mammo_o, dtype = 'float32')/255.
    mammo_f = np.asarray(mammo_f, dtype = 'float32')/255.
    label = np.asarray(label)

    mammo_o_4d = np.reshape(mammo_o, (147, 125, 125, 1))
    mammo_f_4d = np.reshape(mammo_f, (147, 64, 64, 1))

    learning_rate = 0.001
    batch_size = 21
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10,  
                                          verbose=0, mode="auto", min_delta=0.0001, cooldown=0, min_lr=0)

    model_o = nf_cnn()
    model_o.compile(optimizer = SGD(learning_rate, momentum = 0.9), loss = 'binary_crossentropy', metrics = ['accuracy'])
    X_train_o, X_val_o, Y_train_o, Y_val_o = train_test_split(mammo_o_4d, label, test_size = 0.2, random_state = 44)
    
    traino = model_o.fit(X_train_o, Y_train_o,
                     batch_size = batch_size,
                     epochs = 200,
                     verbose=1,
                     validation_data=(X_val_o, Y_val_o),
                     callbacks = [reduce_on_plateau])
    
    



