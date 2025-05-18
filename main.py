import MAB.MAB_e as MAB_e
import MAB.MAB_u as MAB_u
import MAB.MAB_Ts as MAB_Ts
import simulation.simulation as simulation
import pandas as pd
import numpy as np

def load_data(resultat_path: str = 'resultats.csv') -> pd.DataFrame:
    """
    Charge les données de simulation à partir d'un fichier CSV.
    """
    DF = pd.read_csv(resultat_path)
    DF.replace({'N/A': 0}, inplace=True)
    DF = DF.dropna(how='all', subset=['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne'])
    DF[['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne']] = DF[
        ['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne']
    ].apply(pd.to_numeric, errors='coerce')


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