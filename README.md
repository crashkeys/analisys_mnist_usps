# analisys_mnist_usps

Il file contiene il codice scritto su jupyter notebook, nel quale si crea un network e poi si eseguono i seuguenti passi:
- si testa un batch di 100 cifre sulla rete non addestrata: come ci aspettavamo i risultati sono scarsi poiché la rete tenta di indovinare "a caso"
- si addestra una rete sull dataset mnist, tramite la funzione training. 
- adesso testando un batch del dataset mnist l'accuratezza è notevolmente migliorata
- proviamo a testare un batch del dataset usps sulla stessa rete: i pesi e i bias non sono appropriati per questo dataset e la rete ha una scarsa accuratezza
- facciamo di nuovo il training sulla stessa rete ma questa volta considerando il dataset usps: i pesi sono modificati di conseguenza
- il test di un batch di usps adesso è notevolmente migliorato (dal 15% al 90%), ma la performance è peggiorata per il mnist


