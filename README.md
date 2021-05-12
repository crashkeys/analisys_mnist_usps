# Testing catastrophic forgetting and algorithms to mitigate it

# simple fine-tuning
Il file contiene il codice scritto su jupyter notebook, nel quale si crea un network e poi si eseguono i seuguenti passi:
- si testa un batch di 100 cifre sulla rete non addestrata: come ci aspettavamo i risultati sono scarsi poiché la rete tenta di indovinare "a caso"
- si addestra una rete sull dataset mnist, tramite la funzione training. 
- adesso testando un batch del dataset mnist l'accuratezza è notevolmente migliorata
- proviamo a testare un batch del dataset usps sulla stessa rete: i pesi e i bias non sono appropriati per questo dataset e la rete ha una scarsa accuratezza
- facciamo di nuovo il training sulla stessa rete ma questa volta considerando il dataset usps: i pesi sono modificati di conseguenza
- il test di un batch di usps adesso è notevolmente migliorato (dal 15% al 90%), ma la performance è peggiorata per il mnist

# example replay
Vogliamo migliorare la performance del MNIST che dopo il fine-tuning su USPS è peggiorata molto. Viene usato anche Fashion!MNIST per confrontare i risultati ottenuti
- si ripete l'esempio scorso del semplice fine-tuning ma più chiaramente: la rete viene inizialmente addestrata su MNIST e testata sullo stesso e sugli altri due dataset (mnist: 98.8, usps: 44.7, fashion: 11.0) Forward transfer notevole per l'usps, in quanto è ancora un dataset di cifre e quindi simile all'usps, mentre notiamo risultati scarsi per il fashion!mnist
- salviamo la rete in memoria per poterla riutilizzare successivamente
- ripetiamo l'esperimento di fine-tuning su USPS senza memoria. I risultati sono: mnist: 85.47%, usps: 99.45%
- riprendiamo la rete salvata (training su solo mnist) e facciamo il training su USPS, ma questa volta aggiungiamo al dataset degli elementi di MNIST per "rinfrescare la memoria". Dividiamo MNIST in base alle classi ed estraiamo per ciascuna di esse un numero N di esempi random. Questo nuovo dataset costituito da USPS e questi elementi da MNIST viene chiamato "dirty-dataset". 
- Facciamo un training sulla rete di partenza utilizzando dirty-dataset con un numero incrementale di elementi presi da MNIST. 
- Il test di example replay viene fatto per N = 1, 2, 5, 10, 50, 100, 500, 1000, 2000, 5000, e ogni volta viene fatto un test sia su MNIST che su USPS per vedere come sono cambiati i valori dell'accuratezza
- Lo stesso esperimento viene fatto con MNIST e Fashion!MNIST 

# Learning Without Forgetting
Applichiamo la tecnica del LwF al caso di Domain-IL. In un primo passaggio dalla rete addestrata su MNIST vogliamo addestrare su USPS senza dimenticare quello imparato in precedenza. Successivamente a questa rete addestriamo su SVHN, di nuovo con l'obiettivo di non dimenticare i task precedenti

