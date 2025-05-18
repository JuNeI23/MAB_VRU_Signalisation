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



def run_evolution(df: pd.DataFrame, n_arms: int = 3):
    """
    Identique à votre version ε-greedy, mais avec UCB.
    Retourne (times, history_v2v, history_v2i).
    """
    times = sorted(df['Temps'].unique())
    # Préallocation des historiques
    n_times = len(times)
    history_v2v = np.zeros((n_times, n_arms))
    history_v2i = np.zeros((n_times, n_arms))

    mab_v2v = UCBMAB(n_arms)
    mab_v2i = UCBMAB(n_arms)

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
            mab_v2v.update(arm, metrics_v2v[arm])
            mab_v2i.update(arm, metrics_v2i[arm])
            history_v2v[idx, arm] = mab_v2v.values[arm]
            history_v2i[idx, arm] = mab_v2i.values[arm]

    return times, history_v2v, history_v2i

# Fonction pour tracer l'évolution des valeurs UCB
def plot_evolution(df: pd.DataFrame):
    """
    Trace l’évolution des valeurs estimées UCB pour V2V et V2I.
    """

    times, hist_v2v, hist_v2i = run_evolution(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    # Tracer les valeurs UCB pour V2V et V2I

    # Tracer les valeurs UCB pour V2V
    ax1.plot(times, hist_v2v[:, 0], label='Arm 0, Délai moyen (s)')
    ax1.plot(times, hist_v2v[:, 1], label='Arm 1, Taux de perte (%)')
    ax1.plot(times, hist_v2v[:, 2], label='Arm 2, Charge moyenne')
    ax1.set_title('V2V UCB Estimated Values Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('UCB Value')
    ax1.legend()

    # Tracer les valeurs UCB pour V2I
    ax2.plot(times, hist_v2i[:, 0], label='Arm 0, Délai moyen (s)')
    ax2.plot(times, hist_v2i[:, 1], label='Arm 1, Taux de perte (%)')
    ax2.plot(times, hist_v2i[:, 2], label='Arm 2, Charge moyenne')
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
    times, hist_v2v, hist_v2i = run_evolution(df)
    
    # Meilleures valeurs finales
    best_v2v = hist_v2v.max()
    best_v2i = hist_v2i.max()

    # Affichage des résultats
    print(f"Meilleure valeur UCB V2V : {best_v2v:.3f}")
    print(f"Meilleure valeur UCB V2I : {best_v2i:.3f}")
    if best_v2v > best_v2i:
        print("Conclusion : V2V l’emporte.")
    elif best_v2i > best_v2v:
        print("Conclusion : V2I est meilleur.")
    else:
        print("Conclusion : ex æquo.")

