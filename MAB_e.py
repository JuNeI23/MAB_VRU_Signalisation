import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulation.simulation import main

# Lecture unique et nettoyage du CSV
DF = pd.read_csv('resultats.csv').replace('Ø', np.nan)
DF = DF.dropna(subset=['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne'])
DF[['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne']] = DF[
    ['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne']
].apply(pd.to_numeric)

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

    def __str__(self):
        return f"Counts: {self.counts}\nValues: {self.values}"

# Centralized evolution function
def run_evolution(df: pd.DataFrame, epsilon: float, n_arms: int = 3):
    """
    Run ε-greedy MAB updates on V2V and V2I metrics over time.
    Returns (times, history_v2v, history_v2i).
    """
    times = sorted(df['Temps'].unique())
    # Préallocation des historiques
    n_times = len(times)
    history_v2v = np.zeros((n_times, n_arms))
    history_v2i = np.zeros((n_times, n_arms))
    
    # Initialisation des MABs
    eg_v2v = EpsilonGreedyMAB(n_arms, epsilon)
    eg_v2i = EpsilonGreedyMAB(n_arms, epsilon)
    
    # Boucle sur les temps et mise à jour des MABs
    for idx, t in enumerate(times):
        row_v2v = df[(df['Temps'] == t) & (df['Protocole'] == 'V2V')]
        row_v2i = df[(df['Temps'] == t) & (df['Protocole'] == 'V2I')]
        if row_v2v.empty or row_v2i.empty:
            continue
        v2v = row_v2v.iloc[0]
        v2i = row_v2i.iloc[0]
        # Calcul des récompenses vectorisées
        metrics_v2v = np.array([
            v2v['Délai moyen (s)'],
            v2v['Taux de perte (%)'],
            v2v['Charge moyenne']
        ])
        metrics_v2i = np.array([
            v2i['Délai moyen (s)'],
            v2i['Taux de perte (%)'],
            v2i['Charge moyenne']
        ])
        for arm in range(n_arms):
            eg_v2v.update(arm, metrics_v2v[arm])
            eg_v2i.update(arm, metrics_v2i[arm])
            history_v2v[idx, arm] = eg_v2v.values[arm]
            history_v2i[idx, arm] = eg_v2i.values[arm]
    return times, history_v2v, history_v2i

# Fonction pour tracer l'évolution des valeurs ε-greedy
def plot_evolution(epsilon=0.1):
    """
    Plot the evolution of the ε-greedy MAB estimated values over time.
    """
    df = DF

    times, hist_v2v, hist_v2i = run_evolution(df, epsilon)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Plot V2V evolution
    for arm in range(hist_v2v.shape[1]):
        ax1.plot(times, hist_v2v[:, arm], label=f'Arm {arm}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Estimated Value')
    ax1.set_title('V2V MAB Estimated Values Over Time')
    ax1.legend()

    # Plot V2I evolution
    for arm in range(hist_v2i.shape[1]):
        ax2.plot(times, hist_v2i[:, arm], label=f'Arm {arm}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Estimated Value')
    ax2.set_title('V2I MAB Estimated Values Over Time')
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Comparison function
def compare_protocols(epsilon=0.1):
    """
    Compare the final estimated values of the V2V and V2I protocols
    and print which one is better.
    """
    df = DF

    # Run evolution to get histories
    times, hist_v2v, hist_v2i = run_evolution(df, epsilon)

    # Maximum des valeurs estimées
    best_v2v = hist_v2v.max()
    best_v2i = hist_v2i.max()

    # Affichage des résultats
    print(f"Meilleure valeur estimée V2V : {best_v2v:.3f}")
    print(f"Meilleure valeur estimée V2I : {best_v2i:.3f}")
    if best_v2v > best_v2i:
        print("Conclusion : le protocole V2V est meilleur.")
    elif best_v2i > best_v2v:
        print("Conclusion : le protocole V2I est meilleur.")
    else:
        print("Conclusion : les deux protocoles sont équivalents.")


if __name__ == "__main__":
    main()
    # Exécutez les fonctions de traçage et de comparaison
    plot_evolution()
    compare_protocols()