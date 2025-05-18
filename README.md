# MAB_VRU_Signalisation

Une plateforme de simulation et d’analyse de protocoles de communication pour usagers vulnérables (VRU) et véhicules, utilisant des algorithmes Multi-Armed Bandit (MAB) pour choisir dynamiquement la meilleure métrique à optimiser.

---

## 📝 Description du projet

Ce projet se divise en deux parties principales :

1. **Simulation des échanges V2V et V2I**  
   - Modélisation de nœuds (usagers VRU, véhicules, infrastructures)  
   - Transmission de messages suivant différents protocoles (paramétrables)  
   - Collecte de métriques :  
     - Délai de transmission  
     - Taux de perte de paquets  
     - Charge réseau

2. **Sélection dynamique via MAB**  
   - Deux scripts d’analyse qui exploitent les données de simulation (`resultats.csv`) :  
     - **MAB_u.py** : algorithme UCB1 (Upper Confidence Bound)  
     - **MAB_e.py** : algorithme ε-greedy  
   - Ces scripts comparent les performances finales des protocoles V2V et V2I et proposent un classement.

---

## 📂 Structure du projet
MAB_VRU_Signalisation/
├── MAB_u.py               # Analyse UCB des métriques V2V/V2I
├── MAB_e.py               # Analyse ε-greedy des métriques V2V/V2I
├── resultats.csv          # Données brutes de la simulation
└── simulation/
    ├── simulation.py      # Point d’entrée de la simulation (main)
    ├── models.py          # Classes Message, Node, User, Infrastructure
    └── protocols.py       # Classes Protocole et Metric

---

## 🔧 Prérequis

- **Python 3.7+**  
- Bibliothèques Python :
  ```bash 
   pip install numpy pandas matplotlib

## 🚀 Installation & Exécution 
1. Cloner le dépot
   git clone https://votre-repo/MAB_VRU_Signalisation.git
cd MAB_VRU_Signalisation


2. Lancer la simulation
   La simulation génère (ou met à jour) le fichier resultats.csv.
   ```bash
      python simulation/simulation.py

4.	Analyser avec UCB
   ```bash
  	   python MAB_u.py
   ``` 
   •  Affiche la meilleure valeur UCB pour V2V et V2I
	•	Conclusion automatique du protocole gagnant
	•	(Décommentez la ligne plot_evolution() pour visualiser l’évolution)

4.bis	Analyser avec ε-greedy
```bash
   python MAB_e.py
```
   •	Par défaut, ε = 0.1 (modifiable dans l’appel de la fonction)
   •	Affiche la meilleure valeur estimée pour V2V et V2I
   •	(Décommentez plot_evolution(epsilon=…) pour tracer les courbes)

⸻

🔍 Détails des composants

1. simulation/models.py
	•	Message : paquet priorisé, avec timestamp
	•	Node (générique) : géolocalisation, portée, capacité de traitement
	•	User : utilisateur VRU ou véhicule (hérite de Node)
	•	Infrastructure : nœud fixe (hérite de Node)

2. simulation/protocols.py
	•	Protocole :
	•	Paramètres : nom, temps de transmission, taux d’échec, charge réseau
	•	transmit_message() simule délai + perte de paquets
	•	Metric : collecte et calcule
	•	Délai moyen (transmission + file d’attente)
	•	Taux de perte
	•	Charge réseau moyenne

3. MAB_u.py & MAB_e.py
	•	Lecture et nettoyage de resultats.csv
	•	run_evolution() : parcourt chaque instant, met à jour les MABs
	•	compare_protocols() : compare les meilleures récompenses des deux protocoles
	•	plot_evolution() : trace l’historique des valeurs estimées (optionnel)

⸻

⚙️ Personnalisation
	•	Modifier ε (pour ε-greedy) : passez un autre epsilon à plot_evolution() ou compare_protocols().
	•	Nombre de « bras » (métriques) : changez n_arms dans run_evolution() (par défaut 3 : délai, perte, charge).
	•	Paramètres de protocole : dans simulation/protocols.py, ajustez transmission_time, transmission_success_rate, etc.

⸻

📝 Bonnes pratiques
	•	Vérifiez régulièrement la cohérence de resultats.csv (format UTF-8, séparateur par défaut).
	•	Isolez vos tests en modifiant les paramètres dans simulation/simulation.py.
	•	Utilisez un environnement virtuel pour éviter les conflits de dépendances.
