import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GaussianThompsonSampling:
    def __init__(self, n_arms: int):
        """
        Initialise un MAB Thompson Sampling gaussien à n_arms.
        Hypothèse : bruit gaussien de variance unitaire.
        """
        self.n_arms = n_arms
        self.counts = [0] * n_arms
        self.means = [0.0] * n_arms

    def select_arm(self) -> int:
        """
        Sélection selon Thompson Sampling gaussien.
        """
        # Jouer chaque bras au moins une fois
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        # Tirer un échantillon pour chaque bras
        samples = [
            np.random.normal(self.means[arm], 1.0 / np.sqrt(self.counts[arm]))
            for arm in range(self.n_arms)
        ]
        return int(np.argmax(samples))

    def update(self, chosen_arm: int, reward: float):
        """
        Met à jour la moyenne empirique du bras choisi.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.means[chosen_arm]
        # mise à jour incrémentale de la moyenne
        self.means[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward

    def __str__(self):
        return f"Counts: {self.counts}\nMeans: {self.means}"


def run_evolution(df: pd.DataFrame, n_arms: int = 3):
    times = sorted(df['Temps'].unique())
    n_times = len(times)
    history_v2v = np.zeros((n_times, n_arms))
    history_v2i = np.zeros((n_times, n_arms))

    mab_v2v = GaussianThompsonSampling(n_arms)
    mab_v2i = GaussianThompsonSampling(n_arms)

    for idx, t in enumerate(times):
        row_v2v = df[(df['Temps'] == t) & (df['Protocole'] == 'V2V')]
        row_v2i = df[(df['Temps'] == t) & (df['Protocole'] == 'V2I')]
        if row_v2v.empty or row_v2i.empty:
            continue
        v2v = row_v2v.iloc[0]
        v2i = row_v2i.iloc[0]

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
            history_v2v[idx, arm] = mab_v2v.means[arm]
            history_v2i[idx, arm] = mab_v2i.means[arm]

    return times, history_v2v, history_v2i


def plot_evolution(df: pd.DataFrame):
    """
    Trace l'évolution des valeurs de Thompson Sampling pour V2V et V2I.
    """
    times, hist_v2v, hist_v2i = run_evolution(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(times, hist_v2v[:, 0], label='Arm 0, Délai moyen (s)')
    ax1.plot(times, hist_v2v[:, 1], label='Arm 1, Taux de perte (%)')
    ax1.plot(times, hist_v2v[:, 2], label='Arm 2, Charge moyenne')
    ax1.set_title('V2V Thompson Sampling Means Over Time')
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Mean reward')
    ax1.legend()

    ax2.plot(times, hist_v2i[:, 0], label='Arm 0, Délai moyen (s)')
    ax2.plot(times, hist_v2i[:, 1], label='Arm 1, Taux de perte (%)')
    ax2.plot(times, hist_v2i[:, 2], label='Arm 2, Charge moyenne')
    ax2.set_title('V2I Thompson Sampling Means Over Time')
    ax2.set_xlabel('Temps')
    ax2.set_ylabel('Mean reward')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def compare_protocols(df: pd.DataFrame):
    """
    Compare les protocoles V2V et V2I en utilisant Thompson Sampling.
    """

    times, hist_v2v, hist_v2i = run_evolution(df)
    best_v2v = hist_v2v.max()
    best_v2i = hist_v2i.max()

    print(f"Meilleure moyenne Thompson V2V : {best_v2v:.3f}")
    print(f"Meilleure moyenne Thompson V2I : {best_v2i:.3f}")
    if best_v2v > best_v2i:
        print("Conclusion : V2V l’emporte.")
    elif best_v2i > best_v2v:
        print("Conclusion : V2I est meilleur.")
    else:
        print("Conclusion : ex æquo.")
