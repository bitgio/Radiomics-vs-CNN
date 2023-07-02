def plot_train_val(train, title):
    """Si plotta la training e validation acuracy anche la corrispondente loss di entrambi.

    Argomenti
    ----------

    train : 
        Modello allenato.

    title : str
        Titolo dei plots.
    """

    acc = train.history['accuracy']
    val_acc = train.history['val_accuracy']
    loss = train.history['loss']
    val_loss = train.history['val_loss']

    epochs_range = range(1, len(acc)+1)

    t_acc = title + ': training and validation accuracy'
    t_loss = title + ': training and validation loss'
    
    #Train and validation accuracy
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(t_acc)
    
    #Train and validation loss
    plt.subplot(2, 2, 2)    
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(t_loss)
    plt.show()
