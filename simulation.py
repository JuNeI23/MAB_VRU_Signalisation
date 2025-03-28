# Ce fichier appelle un des algorithmes MAB avec le fichier resultats_sumo.csv

Choix = False

while (Choix == False) :

    algorithme_choisi = input("Choisissez un algorithme E ou U")
    match algorithme_choisi:
        case E: