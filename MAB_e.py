import pandas as pd
import numpy as np
import random

# Paramètres
epsilon = 0.1
n_actions = 2  # Direct, Edge Cloud

# Initialisation des valeurs moyennes pour chaque action
action_values = np.zeros(n_actions)
action_counts = np.zeros(n_actions)

def reward_function(row, action):
    """
    Renvoie une récompense simulée selon le mode de communication.
    """
    if action == 0:  # communication directe
        # Favorable si distance faible et bon signal
        return row["signal_quality"] * (1 / (1 + row["distance"]))
    
    elif action == 1:  # communication via Edge Cloud
        # Favorable si infrastructure dispo et réseau peu chargé
        infra_bonus = 0.2 if row["has_infrastructure"] else 0
        return (1 - row["network_load"]) * 0.7 + row["signal_quality"] * 0.2 + infra_bonus

    return 0

def select_action():
    """
    Choix epsilon-greedy de l'action à prendre.
    """
    if random.random() < epsilon:
        return random.randint(0, n_actions - 1)
    else:
        return np.argmax(action_values)

def update(action, reward):
    """
    Met à jour la valeur estimée pour l'action choisie.
    """
    action_counts[action] += 1
    alpha = 1 / action_counts[action]
    action_values[action] += alpha * (reward - action_values[action])

# Chargement du fichier CSV
df = pd.read_csv("simulation.csv")

# Boucle MAB
for _, row in df.iterrows():
    action = select_action()
    reward = reward_function(row, action)
    update(action, reward)

# Résultats
print("Valeurs estimées pour chaque action :", action_values)
print("Nombre de fois où chaque action a été choisie :", action_counts)