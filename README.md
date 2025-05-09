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

---

## 🔧 Prérequis

- **Python 3.7+**  
- Bibliothèques Python :
  ```bash 
  pip install numpy pandas matplotlib
