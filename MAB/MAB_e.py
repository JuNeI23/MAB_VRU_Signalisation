import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Classe pour l'algorithme MAB epsilon-greedy
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
        if np.random.random() < self.epsilon:
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

# Centralized evolution function
def run_evolution(df: pd.DataFrame, epsilon: float, n_arms: int = 2):
    """
    Run ε-greedy MAB updates to select between V2V and V2I protocols over time.
    Returns (times, history) with history shape (n_times, n_arms).
    """
    times = sorted(df['Temps'].unique())
    n_times = len(times)
    eg = EpsilonGreedyMAB(n_arms, epsilon)
    history = np.zeros((n_times, n_arms))

    for idx, t in enumerate(times):
        row_v2v = df[(df['Temps'] == t) & (df['Protocole'] == 'V2V')]
        row_v2i = df[(df['Temps'] == t) & (df['Protocole'] == 'V2I')]
        if row_v2v.empty or row_v2i.empty:
            continue
        v2v = row_v2v.iloc[0]
        v2i = row_v2i.iloc[0]
        # Compute weighted rewards
        reward_v2v = - (0.5 * v2v['Délai moyen (s)'] + 0.3 * v2v['Taux de perte (%)'] + 0.2 * v2v['Charge moyenne'])
        reward_v2i = - (0.5 * v2i['Délai moyen (s)'] + 0.3 * v2i['Taux de perte (%)'] + 0.2 * v2i['Charge moyenne'])

        # Select an arm (0 = V2V, 1 = V2I) and update only that arm
        arm = eg.select_arm()
        if arm == 0:
            eg.update(0, reward_v2v)
        else:
            eg.update(1, reward_v2i)

        # Record estimated values for both arms
        history[idx, :] = eg.values
    return times, history

# Fonction pour tracer l'évolution des valeurs ε-greedy
def plot_evolution(df: pd.DataFrame, epsilon=0.1):
    """
    Plot the evolution of the ε-greedy MAB estimated values over time.
    """
    times, history = run_evolution(df, epsilon)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot V2V evolution (arm 0)
    ax1.plot(times, history[:, 0], label='V2V')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Estimated Value')
    ax1.set_title('V2V MAB Estimated Values Over Time')
    ax1.legend()

    # Plot V2I evolution (arm 1)
    ax2.plot(times, history[:, 1], label='V2I')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Estimated Value')
    ax2.set_title('V2I MAB Estimated Values Over Time')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Comparison function
def compare_protocols(df: pd.DataFrame, epsilon=0.1):
    """
    Compare the final estimated values of the V2V and V2I protocols
    and print which one is better.
    """
    # Run evolution to get histories
    times, history = run_evolution(df, epsilon)

    # Maximum des valeurs estimées
    best_v2v = history[:, 0].max()
    best_v2i = history[:, 1].max()

    # Affichage des résultats
    print(f"Meilleure valeur estimée V2V : {best_v2v:.3f}")
    print(f"Meilleure valeur estimée V2I : {best_v2i:.3f}")
    if best_v2v > best_v2i:
        print("Conclusion : le protocole V2V est meilleur.")
    elif best_v2i > best_v2v:
        print("Conclusion : le protocole V2I est meilleur.")
    else:
        print("Conclusion : les deux protocoles sont équivalents.")
