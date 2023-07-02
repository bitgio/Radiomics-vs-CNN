def f_cnn(shape=(64, 64, 1)):
    """Modelo della CNN ottimizzato per il dataset originale delle mammografie senza filtrare. 
    
    Argomento
    ---------

    shape : tupla
        Dimensionalità delle immagini (64x64 px) con cui si allenarà la CNN e il tipo di *color channels* (in questo caso, trattandosi di immagini in scala di grigi, in questo caso, trattandosi di immagini in scala di grigi, il canale del colore deve essere impostato su 1).
       
    
    Risultato:
        Modelo

    """

    
    model = Sequential([
        
        Conv2D(11, (3,3), padding = 'same', input_shape = shape),
        BatchNormalization(),
        Activation('relu'),
    
        MaxPool2D((4,4), strides = 2),
        
        
        Conv2D(22, (3,3), padding = 'same'),
        BatchNormalization(),
        Activation('relu'),
        
        MaxPool2D((4,4), strides = 2),
        
        
        Conv2D(33, (3,3), padding = 'same'),
        BatchNormalization(),
        Activation('relu'),
        
        MaxPool2D((4,4), strides = 2),
        
        Flatten(),
        
        Dense(10, activation = 'relu'),
        Dense(1, activation = 'sigmoid')        
        
    ])
    
    return model
