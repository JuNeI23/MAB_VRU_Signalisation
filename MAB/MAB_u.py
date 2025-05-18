import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Classe pour l'algorithme MAB epsilon-greedy
class UCBMAB:
    def __init__(self, n_arms: int):
        """
        Initialise un MAB UCB à n_arms.
        """
        self.n_arms = n_arms
        self.counts = [0] * n_arms        # nombre de fois que chaque bras a été joué
        self.values = [0.0] * n_arms      # moyenne empirique des récompenses
        self.total_counts = 0             # nombre total de tours joués

    def select_arm(self) -> int:
        """
        Sélection selon la rule UCB1 :
        """
        # Jouer chaque bras au moins une fois
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # Calculer l’indice UCB pour chaque bras
        ucb_values = [
            self.values[arm] + np.sqrt(2 * np.log(self.total_counts) / self.counts[arm])
            for arm in range(self.n_arms)
        ]
        return int(np.argmax(ucb_values))

    def update(self, chosen_arm: int, reward: float):
        """
        Met à jour la moyenne empirique du bras choisi.
        """
        self.total_counts += 1
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # moyenne incrémentale
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward



def run_evolution(df: pd.DataFrame, n_arms: int = 2):
    """
    Identique à votre version ε-greedy, mais avec UCB.
    Retourne (times, history).
    """
    times = sorted(df['Temps'].unique())
    # Préallocation des historiques
    n_times = len(times)
    mab = UCBMAB(n_arms)
    history = np.zeros((n_times, n_arms))

    for idx, t in enumerate(times):
        row_v2v = df[(df['Temps'] == t) & (df['Protocole'] == 'V2V')]
        row_v2i = df[(df['Temps'] == t) & (df['Protocole'] == 'V2I')]
        if row_v2v.empty or row_v2i.empty:
            continue
        v2v = row_v2v.iloc[0]
        v2i = row_v2i.iloc[0]

        # Calcul de la récompense scalaire pondérée (négative car on veut minimiser)
        reward_v2v = - (0.5 * v2v['Délai moyen (s)'] + 0.3 * v2v['Taux de perte (%)'] + 0.2 * v2v['Charge moyenne'])
        reward_v2i = - (0.5 * v2i['Délai moyen (s)'] + 0.3 * v2i['Taux de perte (%)'] + 0.2 * v2i['Charge moyenne'])

        # Select an arm (0=V2V, 1=V2I) and update that arm
        arm = mab.select_arm()
        if arm == 0:
            mab.update(0, reward_v2v)
        else:
            mab.update(1, reward_v2i)

        # Record estimated values for both arms
        history[idx, :] = mab.values

    return times, history

# Fonction pour tracer l'évolution des valeurs UCB
def plot_evolution(df: pd.DataFrame):
    """
    Trace l’évolution des valeurs estimées UCB pour V2V et V2I.
    """

    times, history = run_evolution(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Tracer les valeurs UCB pour V2V et V2I

    # Tracer les valeurs UCB pour V2V
    ax1.plot(times, history[:, 0], label='V2V')
    ax1.set_title('V2V UCB Estimated Values Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('UCB Value')
    ax1.legend()

    # Tracer les valeurs UCB pour V2I
    ax2.plot(times, history[:, 1], label='V2I')
    ax2.set_title('V2I UCB Estimated Values Over Time')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('UCB Value')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def compare_protocols(df: pd.DataFrame):
    """
    Compare les meilleures valeurs UCB finales de V2V et V2I.
    """

    # Exécutez l'évolution pour obtenir les meilleures valeurs
    times, history = run_evolution(df)
    
    # Meilleures valeurs finales
    best_v2v = history[:, 0].max()
    best_v2i = history[:, 1].max()

    # Affichage des résultats
    print(f"Meilleure valeur UCB V2V : {best_v2v:.3f}")
    print(f"Meilleure valeur UCB V2I : {best_v2i:.3f}")
    if best_v2v > best_v2i:
        print("Conclusion : V2V l’emporte.")
    elif best_v2i > best_v2v:
        print("Conclusion : V2I est meilleur.")
    else:
        print("Conclusion : ex æquo.")
