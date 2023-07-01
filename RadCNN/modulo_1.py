def create_dataset(datfold, l, imo, imf, lab):

    """Funzione che chiama il file di Matlab per filtrare le immagini.
    
    Argumenti
    ---------

    datfold : string
        Percorso della cartella che contiene il dataset.
    
    l : list
        Contiene un pezzo dei percorsi delle immagini del dataset.
    
    imo : list
        Variable dove si accummulano le matrice delle immagine originale (senza filtrare).
    
    imf : list
        Variable dove si accummulano le matrice delle immagine filtrate.

    lab : list
        Variable dove si accummulano i *labels* corrispondenti a ciascune delle matrice.

    
    
    Return:
        Dataset con tutte le immagini originale e filtrate anche i suoi labels.
    """
     
    for element in l:
        if "_1_resized.pgm" in element:
            mo, mf = eng.dataset_filtered(eng.char(os.path.join(datfold, element)), nargout = 2)
            imo.append(mo)
            imf.append(mf)
            lab.append(1.)
        elif "_2_resized.pgm" in element:
            mo, mf = eng.dataset_filtered(eng.char(os.path.join(datfold, element)), nargout = 2)
            imo.append(mo)
            imf.append(mf)
            lab.append(0.)

