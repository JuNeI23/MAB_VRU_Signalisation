import pandas as pd
import numpy as np
import math

def run_ucb_simulation(csv_path):
    # Paramètres
    n_actions = 2  # Direct, Edge Cloud
    c = 2  # Paramètre d'exploration pour UCB

    # Initialisation
    action_values = np.zeros(n_actions)
    action_counts = np.zeros(n_actions)
    total_steps = 0

    def reward_function(row, action):
        """
        Renvoie une récompense simulée selon le mode de communication.
        """
        if action == 0:  # communication directe
            return row["signal_quality"] * (1 / (1 + row["distance"]))
        elif action == 1:  # communication via Edge Cloud
            infra_bonus = 0.2 if row["has_infrastructure"] else 0
            return (1 - row["network_load"]) * 0.7 + row["signal_quality"] * 0.2 + infra_bonus
        return 0

    def select_action_ucb():
        """
        Choix de l'action selon la stratégie Upper Confidence Bound (UCB).
        """
        global total_steps
        total_steps += 1

        # S'assurer que chaque action est essayée au moins une fois
        for a in range(n_actions):
            if action_counts[a] == 0:
                return a

        ucb_values = action_values + c * np.sqrt(np.log(total_steps) / action_counts)
        return np.argmax(ucb_values)

    def update(action, reward):
        """
        Met à jour la valeur estimée pour l'action choisie.
        """
        action_counts[action] += 1
        alpha = 1 / action_counts[action]
        action_values[action] += alpha * (reward - action_values[action])

    # Chargement du fichier CSV
    df = pd.read_csv(csv_path)

    # Boucle MAB UCB
    for _, row in df.iterrows():
        action = select_action_ucb()
        reward = reward_function(row, action)
        update(action, reward)

    # Résultats
    return action_values.tolist()