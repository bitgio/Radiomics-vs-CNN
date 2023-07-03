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

In questo Python file si può trovare l'implementazione della CNN ottimizzata con il dataset con il filtro.


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
from keras.preprocessing.image import ImageDataGenerator


__all__ = ['create_f_dataset', 'f_cnn', 'f_cnn_aug', 'plot_train_val']

from modulo_1 import create_f_dataset
from modulo_2 import f_cnn
from modulo_3 import f_cnn_aug
from modulo_4 import plot_train_val



if __name__ == '__main__':

    mammo_f, label = [], []
    data_folder = "../../dataset_filtered/"
    os.chdir(data_folder)
    listdir = os.listdir()
    threads = []
    chunk = 6

    for i in range(49):
        t = thr.Thread(target = create_f_dataset, args = (data_folder, listdir[i*chunk : (i+1)*chunk], mammo_f, label))
        threads.append(t)
        t.start()
    
    for j in threads:
        j.join()


    mammo_f = np.asarray(mammo_f, dtype = 'float32')/255.
    label = np.asarray(label)
    mammo_f_4d = np.reshape(mammo_f, (147, 64, 64, 1))

    learning_rate = 0.0005
    batch_size = 21
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10,  
                                          verbose=0, mode="auto", min_delta=0.0001, cooldown=0, min_lr=0)


    # MODELLO ALLENATO CON LE IMMAGINI FILTRATE
    model_f = f_cnn()
    model_f.compile(optimizer = SGD(learning_rate, momentum = 0.9), loss = 'binary_crossentropy', metrics = ['accuracy'])
    X_train_f, X_val_f, Y_train_f, Y_val_f = train_test_split(mammo_f_4d, label, test_size = 0.2, random_state = 44)

    train = model_f.fit(X_train_f, Y_train_f,
                     batch_size = batch_size,
                     epochs = 200,
                     verbose=1,
                     validation_data=(X_val_f, Y_val_f),
                     callbacks = [reduce_on_plateau])
    
    plot_train_val(train, 'Filtered Dataset')



    # MODELLO ALLENATO CON LE IMMAGINI FILTRATE E QUELLE CHE SONO STATE CREATE CON LA DATA AUGMENTATION  
    model_f_aug = f_cnn_aug()
    aug = ImageDataGenerator(
                rotation_range = 90,
                horizontal_flip = True,
                vertical_flip = True,
                validation_split = 0.20)

    aug_train_f = aug.flow(mammo_f_4d, label, batch_size = 30, subset = 'training')
    aug_val_f = aug.flow(mammo_f_4d, label, batch_size = 30, subset = 'validation')

    train_aug = model_f_aug.fit(aug_train_f,
                     batch_size = batch_size,
                     epochs = 200,
                     verbose=1,
                     validation_data=(aug_val_f),
                     callbacks = [reduce_on_plateau])
    
    plot_train_val(train_aug, 'Filtered Augmented Dataset')



    # CONFRONTO DELLA PERFORMANCE TRA IL MODELLO USANDO O NO IL DATA AUGMENTATION
    acc_f = []
    acc_f_aug = []
    for j in range(10):
        _, val_acc = model_f.evaluate(X_val_f, Y_val_f, verbose=0)
        print('Validation accuracy: %.3f' % (val_acc))
        acc_f.append(val_acc)
        _, val_acc = model_f_aug.evaluate(aug_val_f, verbose=0)
        print('Validation accuracy: %.3f' % (val_acc))
        acc_f_aug.append(val_acc)
    
    print(f'DATASET FILTERED:\nMean validation accuracy: {np.mean(acc_f)} \tMean validation std: {np.std(acc_f)}')
    print(f'DATASET AUGMENTED FILTERED:\nMean validation accuracy: {np.mean(acc_f_aug)} \tMean validation loss: {np.std(acc_f_aug)}')



    # TESTING IL MODELLO PIÙ AFFIDABILE
    mammo_f_t, label_t = [], []
    data_folder_t = "../test_dataset/"
    os.chdir(data_folder_t)
    l_t = os.listdir()

    os.chdir("./")
    threads = []
    chunk = 6

    for i in range(10):
        t = thr.Thread(target = create_f_dataset, args = (data_folder_t, l_t[i*chunk : (i+1)*chunk], mammo_f_t, label_t))
        threads.append(t)
        t.start()
        
    for j in threads:
        j.join()

    mammo_f_t = np.asarray(mammo_f_t, dtype = 'float32')/255.
    label_t = np.asarray(label_t)
    mammo_f_4d_t = np.reshape(mammo_f_t, (30, 64, 64, 1))

    if np.mean(acc_f) > np.mean(acc_f_aug):
        test_loss, test_acc = model_f.evaluate(mammo_f_4d_t, label_t)
        preds_test = model_f.predict(mammo_f_4d_t, verbose=1)
        fpr, tpr, _ = roc_curve(label_t, preds_test)
        roc_auc = auc(fpr, tpr)

        print('\n Test accuracy = %.3f'% (test_acc))
        print('\n AUC = %.3f'% (roc_auc))
    else:
        test_loss, test_acc = model_f_aug.evaluate(mammo_f_4d_t, label_t)
        preds_test = model_f_aug.predict(mammo_f_4d_t, verbose=1)
        fpr, tpr, _ = roc_curve(label_t, preds_test)
        roc_auc = auc(fpr, tpr)
        
        print('\n Test accuracy = %.3f'% (test_acc))
        print('\n AUC = %.3f'% (roc_auc))
