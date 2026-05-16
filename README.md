L'obbiettivo è quello di combinare la mappatura tramite prm con l'APF, una volta trovato il percorso tramite prm una
formazione ad anello formata da un'agente centrale e n satelliti si muove seguendo i waypoint che formano il percorso
trovato tramite prm.


L'agente centrle è l'unico a percepire il potenziale attrattivo dei waypoint, i satelliti percepiscono
vari potenziali repulsivi necessari a mantenere la formazione ed evitare collisioni. Il sistema permette di scalare dimensione di immagine, ostacoli e raggio dell'orbita in base al numero di agenti.


L'algoritmo utilizzato per la navigazione è il prm con l'implementazione dei k neighbors. Questo algoritmo genera una serie di punti casuali, verifica che i punti non si trovino all'interno di un ostacolo e collega ciascun punto con i K nodi più vicini che possono essere collegati. Il percorso viene  trovato utilizzando la funzione nx.shortest-path che utilizza l'algoritmo di dijkstra.
Tramite la variabile MIN\_START\_GOAL\_DIST si può impostare la distanza minima tra inizio e fine del percorso, 
anche questa distanza viene scalata in base al numero di agenti.
Gli ostacoli sono di forma rettangolare, la loro dimensione esatta è randomica, la loro posizione è randomica con il vincolo
di non sovrapporsi. Il numero di ostacoli è variabile e difinito dalla variabile NUM\_OBSTACLES.
I parametri dell'algoritmo prm sono il numero di nodi vicini da collegare, il numero di nodi minimi d generare e il numero di nodi massimi.


I potenziali artificiali usati comprendono:
Un potenziale attrattivo posizionato nel prossimo nodo del percorso che viene percepito solo dall'agente centrale 
Un potenziale repulsivo di ciascun agente percepito da tutti gli agenti per evitare le collisioni
Un potenziale repulsivo degli ostacoli per evitare le collisioni, tutti gli agenti percepiscono una forza normale rispetto
all'ostacolo e una forza tangenziale direzionata verso l'agente centrale
Un potenziale che vincola i satelliti a mantenere una formazione, rappresentano i lati e le diagonali della formazione
a poligono, con intensità inversamente proporzionale alla lunghezza
Due potenziali repulsivi percepiti dai satelliti, uno più debole posizionato all'interno dell'orbita e uno più forte
posizionato all'esterno, mantengono i satelliti attorno all'agente centrale


L'agente centrale si muove verso il prossimo waypoint, si ferma per un certo periodo su ogni waypoint in base al valore
di WAIT\_TIME\_SECONDS e quando arriva all'ultimo rimane fermo. I satelliti vengono mantenuti nella giusta posizione dai
potenziali repulsivi interni ed esterni della circonferenza e dai vincoli elastici, viene inoltre utilizzato un feedforward
relativo alla velocità dell'agente centrale per evitare che i satelliti rimangano troppo indietro.
I satelliti idealmente dovrebbero rimanere nell'orbita attorno all'agente centrale disposti in maniera uniforme, 
quindi dovrebbero disporsi a formare un poligono regolare inscritto nell'orbita. 
La distanza reciproca che dovrebbero mantenere può essere calcolata a partire dal numero di satelliti, questo viene fatto ad ogni waypoint raggiunto, se non sono troppo vicini, per riassegnare i vincoli in modo che gli agenti non rimangano incastrati. 


Quando la formazione arriva al goal finale i vincoli vengono ricalcolari periodicamente. 
La variabile MAX\_DIAG\_STEPS indica quante diagonali vengono effettivamente considerate per ogni satellite da entrambi 
i lati, se MAX\_DIAG\_STEPS è settata a 0 non verranno considerate diagonali ma solo i lati del poligono.
Nel caso in cui un satellite rimanga incastrato in un minimo locale viene trovato il nodo del grafo più vicino al satellite
bloccato, quello più vicino alla posizione attuale dell'agente centrale e viene calcolato un percorso che li collega.
Il satellite comincia a seguire questo percorso e mentre lo fa ignora i potenziali artificiali relativi ai vincoli elastici
e all'orbita. Nel caso rimanga nuovamente bloccato mentre segue questo percorso ne viene calcolato un altro.
Nel caso in cui il nodo più vicino al satellite appartenga ad una componente connessa separata e non si possa trovare un
percorso farà un movimento casuale in una direzione libera.
