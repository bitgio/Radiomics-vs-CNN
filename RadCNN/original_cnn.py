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
