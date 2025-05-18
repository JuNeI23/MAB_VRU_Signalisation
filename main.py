import MAB.MAB_e as MAB_e
import MAB.MAB_u as MAB_u
import MAB.MAB_Ts as MAB_Ts
import simulation.simulation as simulation
import pandas as pd
import numpy as np

def load_data(v2v_path: str = 'resultats_V2V.csv', v2i_path: str = 'resultats_V2I.csv') -> pd.DataFrame:
    """
    Charge et nettoie les résultats V2V et V2I, puis renvoie un DataFrame fusionné.
    """
    DF_V2V = pd.read_csv(v2v_path).replace({'N/A', np.nan})
    DF_V2V = DF_V2V.dropna(how='all', subset=['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne'])
    DF_V2V[['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne']] = DF_V2V[
        ['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne']
    ].apply(pd.to_numeric, errors='coerce')

    DF_V2I = pd.read_csv(v2i_path).replace({'N/A': np.nan})
    DF_V2I = DF_V2I.dropna(how='all', subset=['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne'])
    DF_V2I[['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne']] = DF_V2I[
        ['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne']
    ].apply(pd.to_numeric, errors='coerce')

    DF = pd.merge(DF_V2V, DF_V2I, on='Temps', suffixes=('_V2V', '_V2I'))
    return DF

def simu():
    # Simulation de la communication
    print("[Étape] Simulation de la communication...")
    simulation.main()
    
    # Chargement des données
    print("[Étape] Chargement des données...")
    df = load_data()

    # Comparaison des protocoles
    print("[Étape] Comparaison des protocoles...")
    MAB_e.compare_protocols(df)
    MAB_u.compare_protocols(df)
    MAB_Ts.compare_protocols(df)

    # Tracer l'évolution des protocoles
    print("[Étape] Tracer l'évolution des protocoles...")
    MAB_e.plot_evolution(df)
    MAB_u.plot_evolution(df)
    MAB_Ts.plot_evolution(df)
    print("Simulation terminée.")
    print("Comparaison des protocoles terminée.")

if __name__ == "__main__":
    simu()