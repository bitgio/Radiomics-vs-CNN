def create_nf_dataset(datfold, l, imo, lab):

    """Funzione legge il dataset delle immagini senza il filtro.
    
    Argomenti
    ---------

    datfold : string
        Percorso della cartella che contiene il dataset.
    
    l : list
        Contiene un pezzo dei percorsi delle immagini del dataset.
    
    imo : list
        Variable dove si accummulano le matrice delle immagine originale (senza filtrare).
    
    lab : list
        Variable dove si accummulano i *labels* corrispondenti a ciascune delle matrice.
    
    
    Risultato:
        Dataset con tutte le immagini originale senza filtrare anche i suoi labels.
    """
     
    for element in l:
        if "_1_resized.pgm" in element:
            mo = imread(os.path.join(datfold, element))
            imo.append(mo)
            lab.append(1.)
        elif "_2_resized.pgm" in element:
            mo = imread(os.path.join(datfold, element))
            imo.append(mo)
            lab.append(0.)
