INSTALLAZIONE
=============

In questo file viene spiegata la procedura per inizializare il progetto.

Git Repository
--------------



Il presente progetto è stato sviluppato con il sistemma di VSC cosidetto *git*. In particolare, *GitHuB* viene usato come il *hosting server* preferito.



Si deve accedere al *repository* `Radiomics vs CNN <https://github.com/bitgio/Radiomics-vs-CNN.git>`_. È un *repository* pubblico visibile a tutti.      
Prima di esecutare il codice è necessario clonare il *repository* affinché la vostra *working copy* abbia anche il dataset utilizzato.                  
Una volta fatto questo, sarà possibile esecutare il codice principale.                                                                                  



Importing packages
-------------------



Nel *repository* si trova il file di testo *requirements.txt* in cui specificano le librerie di Python usate. Si consiglia avere una                   
versione uguale o superiore a Python 3.9.                                                                                                              



Inoltre, la parte relativa alla radiomica è stata scritta in linguaggio Matlab. Pertanto è necessario disporre della piattaforma Matlab. Si raccomanda 
vivamente di aggiornare la licenza alla versione Matlab 2023a per evitare errori di esecuzione nelle funzioni e nei *Matlab Toolbox* usati.            



Code execution
---------------


Il codice si trova nella cartella chiamata `RadCNN <https://github.com/bitgio/Radiomics-vs-CNN/tree/main/RadCNN>*`_. È diviso in due sezioni: la parte
con la pipeline radiomica e quella della CNN.

* radiomica:
    Si trova il file di Matlab dove viene implementata tutta la pipeline della radiomica. L'esecuzione di questo script ritorna anche il dataset delle mammografie
    in cui si ha usato il filtro della wavelet decomposition.

* CNN:
    Dentro di questa cartella prima si trovano i Jupyter Notebooks dove è stato implementato la CNN per i due casi che sono stati confrontati: mammografie originale
    e mammografie filtrate con la wavelet decomposition di Matlab. In ciascuno dei notebooks discussi in precedenza viene messo anche il confronto tra i modelli delle CNN
    allenate soltanto con i dataset originale oppure con un dataset più grande (cioè, con un set di dati aumentati tramitte l'uso della data augmentation).


    Ci sono anche delle sottocartelle dove si è spezzato il codice per potere compilare del modo corretto la documentazione.
