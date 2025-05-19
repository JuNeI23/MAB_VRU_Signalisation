import MAB.MAB_e as MAB_e
import MAB.MAB_u as MAB_u
import MAB.MAB_Ts as MAB_Ts
import simulation.simulation as simulation
import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
def setup_logging() -> None:
    """Configure the logging system with console and file handlers."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    
    # File handler with detailed formatting
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

setup_logging()
logger = logging.getLogger(__name__)

def load_data(resultat_path: str = 'resultats.csv') -> pd.DataFrame:
    """
    Charge les données de simulation à partir d'un fichier CSV.
    
    Args:
        resultat_path (str): Chemin vers le fichier CSV contenant les résultats
        
    Returns:
        pd.DataFrame: DataFrame contenant les données nettoyées
        
    Raises:
        FileNotFoundError: Si le fichier n'existe pas
        pd.errors.EmptyDataError: Si le fichier est vide
        ValueError: Si les colonnes requises sont manquantes ou les données invalides
    """
    logger.info(f"Chargement des données depuis {resultat_path}")
    try:
        DF = pd.read_csv(resultat_path)
        logger.debug(f"Fichier chargé avec succès: {len(DF)} lignes")
        
        if DF.empty:
            logger.error("Le fichier CSV est vide")
            raise pd.errors.EmptyDataError("Le fichier CSV est vide")
            
        # Validate required columns
        required_columns = ['Time', 'Protocol', 'Average Delay (s)', 'Loss Rate (%)', 'Average Load']
        missing_columns = [col for col in required_columns if col not in DF.columns]
        if missing_columns:
            logger.error(f"Colonnes manquantes: {missing_columns}")
            raise ValueError(f"Colonnes requises manquantes: {', '.join(missing_columns)}")
            
        # Rename columns to French for compatibility with existing code
        column_mapping = {
            'Time': 'Temps',
            'Protocol': 'Protocole',
            'Average Delay (s)': 'Délai moyen (s)',
            'Loss Rate (%)': 'Taux de perte (%)',
            'Average Load': 'Charge moyenne'
        }
        DF.rename(columns=column_mapping, inplace=True)
        logger.debug("Colonnes renommées en français")
            
        # Fix: replace 'N/A' with np.nan instead of 0 for better data handling
        DF.replace({'N/A': np.nan}, inplace=True)
        logger.debug("Valeurs N/A remplacées par NaN")
        
        # Fix: drop rows where all metric columns are NaN
        metric_columns = ['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne']
        initial_rows = len(DF)
        DF = DF.dropna(how='all', subset=metric_columns)
        dropped_rows = initial_rows - len(DF)
        if dropped_rows > 0:
            logger.warning(f"{dropped_rows} lignes supprimées car toutes les métriques étaient NaN")
        
        # Fix: convert to numeric, keeping NaN values
        DF[metric_columns] = DF[metric_columns].apply(pd.to_numeric, errors='coerce')
        logger.debug("Conversion des colonnes métriques en valeurs numériques")
        
        # Validate protocols
        valid_protocols = ['V2V', 'V2I']
        invalid_protocols = DF[~DF['Protocole'].isin(valid_protocols)]['Protocole'].unique()
        if len(invalid_protocols) > 0:
            logger.error(f"Protocoles invalides détectés: {invalid_protocols}")
            raise ValueError(f"Protocoles invalides détectés: {', '.join(invalid_protocols)}")
            
        # Log metric ranges for debugging
        for col in ['Délai moyen (s)', 'Taux de perte (%)', 'Charge moyenne']:
            min_val = DF[col].min()
            max_val = DF[col].max()
            logger.debug(f"{col}: min={min_val}, max={max_val}")
            
        # Validate that we have enough data after cleaning
        if len(DF) == 0:
            logger.error("Aucune donnée valide après nettoyage")
            raise ValueError("Aucune donnée valide après nettoyage")
            
        logger.info(f"Données chargées et nettoyées avec succès: {len(DF)} lignes valides")
        return DF
        
    except FileNotFoundError:
        logger.error(f"Fichier non trouvé: {resultat_path}")
        raise FileNotFoundError(f"Impossible de trouver le fichier: {resultat_path}")
    except pd.errors.EmptyDataError:
        logger.error(f"Fichier vide: {resultat_path}")
        raise pd.errors.EmptyDataError(f"Le fichier {resultat_path} est vide")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {str(e)}", exc_info=True)
        raise Exception(f"Erreur lors du chargement des données depuis {resultat_path}: {str(e)}")

def simu() -> bool:
    """
    Exécute la simulation et l'analyse des protocoles de communication.
    
    Cette fonction orchestre l'ensemble du processus de simulation:
    1. Exécute la simulation de communication
    2. Charge et valide les données de simulation
    3. Compare les protocoles avec différents algorithmes MAB
    4. Trace l'évolution des performances des protocoles
    
    Returns:
        bool: True si la simulation s'est terminée avec succès, False sinon
    """
    try:
        # Simulation de la communication
        logger.info("[Étape 1/4] Démarrage de la simulation de communication")
        if not simulation.main():
            logger.error("La simulation a échoué")
            raise RuntimeError("La simulation a échoué")
        
        # Chargement des données
        logger.info("[Étape 2/4] Chargement et validation des données")
        df = load_data()
        
        if df.empty:
            logger.error("Aucune donnée valide n'a été chargée")
            raise ValueError("Aucune donnée valide n'a été chargée")

        # Comparaison des protocoles
        logger.info("[Étape 3/4] Comparaison des protocoles")
        
        logger.info("Analyse avec Epsilon-Greedy:")
        MAB_e.compare_protocols(df)
        
        logger.info("Analyse avec UCB:")
        MAB_u.compare_protocols(df)
        
        logger.info("Analyse avec Thompson Sampling:")
        MAB_Ts.compare_protocols(df)

        # Tracer l'évolution des protocoles
        logger.info("[Étape 4/4] Génération des graphiques d'évolution")
        MAB_e.plot_evolution(df)
        MAB_u.plot_evolution(df)
        MAB_Ts.plot_evolution(df)
        
        logger.info("Simulation terminée avec succès")
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Fichier non trouvé: {str(e)}")
    except pd.errors.EmptyDataError:
        logger.error("Le fichier de résultats est vide")
    except ValueError as e:
        logger.error(f"Erreur de validation: {str(e)}")
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la simulation: {str(e)}", exc_info=True)
    
    return False

if __name__ == "__main__":
    success = simu()
    if not success:
        logger.error("La simulation s'est terminée avec des erreurs")