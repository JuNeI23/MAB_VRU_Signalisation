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


def run_evolution(df: pd.DataFrame, n_arms: int = 2):
    times = sorted(df['Temps'].unique())
    n_times = len(times)
    mab = GaussianThompsonSampling(n_arms)
    history = np.zeros((n_times, n_arms))

    for idx, t in enumerate(times):
        row_v2v = df[(df['Temps'] == t) & (df['Protocole'] == 'V2V')]
        row_v2i = df[(df['Temps'] == t) & (df['Protocole'] == 'V2I')]
        if row_v2v.empty or row_v2i.empty:
            continue
        v2v = row_v2v.iloc[0]
        v2i = row_v2i.iloc[0]

        reward_v2v = - (0.5 * v2v['Délai moyen (s)'] + 0.3 * v2v['Taux de perte (%)'] + 0.2 * v2v['Charge moyenne'])
        reward_v2i = - (0.5 * v2i['Délai moyen (s)'] + 0.3 * v2i['Taux de perte (%)'] + 0.2 * v2i['Charge moyenne'])

        # Select a protocol arm (0=V2V, 1=V2I) and update only that arm
        arm = mab.select_arm()
        if arm == 0:
            mab.update(0, reward_v2v)
        else:
            mab.update(1, reward_v2i)

        # Record the current estimated means for both arms
        history[idx, :] = mab.means

    return times, history


def plot_evolution(df: pd.DataFrame):
    """
    Trace l'évolution des valeurs de Thompson Sampling pour V2V et V2I.
    """
    times, history = run_evolution(df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(times, history[:, 0], label='V2V')
    ax1.set_title('V2V Thompson Sampling Mean Over Time')
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Mean reward')
    ax1.legend()

    ax2.plot(times, history[:, 1], label='V2I')
    ax2.set_title('V2I Thompson Sampling Mean Over Time')
    ax2.set_xlabel('Temps')
    ax2.set_ylabel('Mean reward')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def compare_protocols(df: pd.DataFrame):
    """
    Compare les protocoles V2V et V2I en utilisant Thompson Sampling.
    """

    times, history = run_evolution(df)
    best_v2v = history[:, 0].max()
    best_v2i = history[:, 1].max()

    print(f"Meilleure moyenne Thompson V2V : {best_v2v:.3f}")
    print(f"Meilleure moyenne Thompson V2I : {best_v2i:.3f}")
    if best_v2v > best_v2i:
        print("Conclusion : V2V l’emporte.")
    elif best_v2i > best_v2v:
        print("Conclusion : V2I est meilleur.")
    else:
        print("Conclusion : ex æquo.")
