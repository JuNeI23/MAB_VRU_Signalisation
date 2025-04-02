import numpy as np
import pandas as pd

class EpsilonGreedyMAB:
    def __init__(self, n_arms, epsilon):
        """
        Initialise l'algorithme MAB epsilon-greedy.

        :param n_arms: nombre de bras (actions possibles)
        :param epsilon: probabilité d'exploration (0 <= epsilon <= 1)
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = [0] * n_arms        # nombre de fois que chaque bras a été choisi
        self.values = [0.0] * n_arms      # valeur estimée de chaque bras

    def select_arm(self):
        """
        Sélectionne un bras selon la stratégie epsilon-greedy.

        :return: index du bras sélectionné
        """
        if np.random() < self.epsilon:
            # Exploration : choisir un bras au hasard
            return np.random.randint(self.n_arms)
        else:
            # Exploitation : choisir le bras avec la meilleure valeur estimée
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        """
        Met à jour les statistiques après avoir joué un bras.

        :param chosen_arm: index du bras joué
        :param reward: récompense obtenue
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # Mise à jour incrémentale de la moyenne
        self.values[chosen_arm] = ((n - 1) / float(n)) * value + (1 / float(n)) * reward

    def __str__(self):
        return f"Counts: {self.counts}\nValues: {self.values}"
    
#Chargement des données
data = pd.read_csv('data.csv',header=None)

# Extraction des données V2V et V2I
# Extraction des données V2V
v2v_delay = data.iloc[0, 0]
v2v_loss_rate = data.iloc[0, 1]
v2v_load = data.iloc[0, 2]

# Extraction des données V2I
v2i_delay = data.iloc[1, 0]
v2i_loss_rate = data.iloc[1, 1]
v2i_load = data.iloc[1, 2]


#Inversion des valeurs pour la simulation et pondération
delay_weight = 1   
loss_rate_weight = 1
load_weight = 0.2

# Initialisation de l'algorithme MAB pour V2V et V2I
epsilon = 0.1
eg_mab_v2v = EpsilonGreedyMAB(n_arms=3, epsilon=epsilon)
eg_mab_v2i = EpsilonGreedyMAB(n_arms=3, epsilon=epsilon)

# Mise à jour des bras avec les données V2V
eg_mab_v2v.update(0, v2v_delay)
eg_mab_v2v.update(1, v2v_loss_rate)
eg_mab_v2v.update(2, v2v_load)

# Mise à jour des bras avec les données V2I
eg_mab_v2i.update(0, v2i_delay)
eg_mab_v2i.update(1, v2i_loss_rate)
eg_mab_v2i.update(2, v2i_load)

# Affichage des résultats
print("V2V MAB Results:")
print(eg_mab_v2v)
print("\nV2I MAB Results:")
print(eg_mab_v2i)
# Affichage des bras sélectionnés
print("\nV2V Selected Arm:", eg_mab_v2v.select_arm())
print("V2I Selected Arm:", eg_mab_v2i.select_arm())
# Affichage des valeurs estimées
print("\nV2V Estimated Values:", eg_mab_v2v.values)
print("V2I Estimated Values:", eg_mab_v2i.values)
# Affichage des compteurs
print("\nV2V Counts:", eg_mab_v2v.counts)
print("V2I Counts:", eg_mab_v2i.counts)

