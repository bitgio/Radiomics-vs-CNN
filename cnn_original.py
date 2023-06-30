import os
import numpy as np
import threading as thr
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import roc_curve, auc
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, InputLayer, Activation, Dropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD
import matlab.engine

def create_dataset(lista, o_img, f_img, labels):
    """Function calling the Matlab file in order to filter the images.
    
    Arguments
    ---------
    
    lista : list
        Chunk of file directories.
    
    Return:
        Dataset with all the images filtered.
    """
     
    for element in lista:
        if "_1_resized.pgm" in element:
            mo, mf = eng.dataset_filtered(eng.char(os.path.join(data_folder, element)), nargout = 2)
            o_img.append(mo)
            f_img.append(mf)
            labels.append(1.)
        elif "_2_resized.pgm" in element:
            mo, mf = eng.dataset_filtered(eng.char(os.path.join(data_folder, element)), nargout = 2)
            o_img.append(mo)
            f_img.append(mf)
            labels.append(0)



def cnn_o(shape=(125, 125, 1)):
    """CNN original mammo model.
    
    Arguments
    ---------
    
    shape : tuple
        Size.
    
    Return:
        Dataset with all the images filtered.
    """
    
    model = Sequential([
        
        Conv2D(4, (3,3), padding = 'same', input_shape = shape),
        BatchNormalization(),
        Activation('relu'),
        
        MaxPool2D((6,6), strides = 2),
        #Dropout(0.2),
        
        
        Conv2D(7, (3,3), padding = 'same'),
        BatchNormalization(),
        Activation('relu'),
        
        MaxPool2D((6,6), strides = 2),
        #Dropout(0.1),
        
        
        Conv2D(10, (3,3), padding = 'same'),
        BatchNormalization(),
        Activation('relu'),
        
        MaxPool2D((6,6), strides = 2),
        #Dropout(0.1),
        
        Flatten(),
        
        Dense(10, activation = 'relu'),
        #Dropout(0.1),
        Dense(1, activation = 'sigmoid')        
        
    ])
    
    return model


if __name__ == '__main__':
    eng = matlab.engine.start_matlab()

    mammo_o, mammo_f, label = [], [], []
    data_folder = "C:/Users/anapascual/exam_project/dataset/"
    os.chdir(data_folder)
    l = os.listdir()

    os.chdir("C:/Users/anapascual/exam_project/")
    threads = []
    chunk = 6

    for i in range(49):
        t = thr.Thread(target = create_dataset, args = (l[i*chunk : (i+1)*chunk], mammo_o, mammo_f, label))
        threads.append(t)
        t.start()
        
    for j in threads:
        j.join()

    eng.quit()

    mammo_o = np.asarray(mammo_o, dtype = 'float32')/255.
    mammo_f = np.asarray(mammo_f, dtype = 'float32')/255.
    label = np.asarray(label)

    mammo_o_4d = np.reshape(mammo_o, (147, 125, 125, 1))
    mammo_f_4d = np.reshape(mammo_f, (147, 64, 64, 1))

    model_o = cnn_o()
    model_o.summary()

    learning_rate = 0.001
    model_o.compile(optimizer = SGD(learning_rate, momentum = 0.9), loss = 'binary_crossentropy', metrics = ['accuracy'])

    reduce_on_plateau = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=10,
    verbose=0,
    mode="auto",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0)

    X_train_o, X_val_o, Y_train_o, Y_val_o = train_test_split(mammo_o_4d, label, test_size = 0.2, random_state = 44)
    batch_size = 21
    traino = model_o.fit(X_train_o, Y_train_o, 
                     batch_size = batch_size,
                     epochs = 200, 
                     verbose=1,
                     validation_data=(X_val_o, Y_val_o),
                     callbacks = [reduce_on_plateau])
    
    acc = traino.history['accuracy']
    val_acc = traino.history['val_accuracy']
    loss = traino.history['loss']
    val_loss = traino.history['val_loss']

    epochs_range = range(1, len(acc)+1)

    #Train and validation accuracy 
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    #Train and validation loss 
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    _, val_acc = model_o.evaluate(X_val_o, Y_val_o, verbose=0)
    print('Validation accuracy: %.3f' % (val_acc))

    preds = model_o.predict(X_val_o, verbose=1)

    #Compute Receiver operating characteristic (ROC)
    fpr, tpr, _ = roc_curve(Y_val_o, preds)
    roc_auc = auc(fpr, tpr)

    #Plot of a ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
