from typing import List, Dict
import pandas as pd
import csv

from simulation.models import User, Infrastructure, Node
from simulation.protocols import Protocole
from simulation.metric import Metric

def charger_usagers_depuis_csv(fichier_csv: str) -> List[Node]:
    print(f"[Étape] Chargement du fichier CSV : {fichier_csv}")
    df = pd.read_csv(fichier_csv)
    usagers: List[Node] = []
    for _, row in df.iterrows():
        t = row['_time']
        if pd.notna(row.get('person/1/_id')):
            proto = Protocole("V2I", network_load=0.1, packet_loss_rate=0.05, transmission_time=0.5)
            infra = Infrastructure(
                id=row['person/1/_id'],
                protocol=proto,
                x=row['person/1/_x'],
                y=row['person/1/_y'],
                processing_capacity=100,
                time=t
            )
            usagers.append(infra)
        if pd.notna(row.get('person/0/_id')):
            user = User(
                usager_id=row['person/0/_id'],
                x=row['person/0/_x'],
                y=row['person/0/_y'],
                angle=row['person/0/_angle'],
                speed=row['person/0/_speed'],
                position=row['person/0/_pos'],
                lane=row['person/0/_edge'],
                time=t,
                usager_type=row['person/0/_type'],
                categorie="vru"
            )
            usagers.append(user)
        i = 0
        while f"vehicle/{i}/_id" in row:
            vid = row[f"vehicle/{i}/_id"]
            if pd.notna(vid):
                usager = User(
                    usager_id=vid,
                    x=row[f"vehicle/{i}/_x"],
                    y=row[f"vehicle/{i}/_y"],
                    angle=row[f"vehicle/{i}/_angle"],
                    speed=row[f"vehicle/{i}/_speed"],
                    lane=row[f"vehicle/{i}/_lane"],
                    position=row[f"vehicle/{i}/_pos"],
                    time=t,
                    usager_type=row[f"vehicle/{i}/_type"],
                    categorie="vehicule"
                )
                usagers.append(usager)
            i += 1
    return usagers

def regrouper_par_temps(usagers: List[Node]) -> Dict[float, List[Node]]:
    print("[Étape] Regroupement des usagers par temps")
    result: Dict[float, List[Node]] = {}
    for u in usagers:
        result.setdefault(u.time, []).append(u)
    return result

def simuler_communication(users: List[Node], protocole: Protocole, metric: Metric, mode: str = "v2v") -> Metric:
    print(f"[Étape] Simulation de communication (mode = {mode})")
    if mode == "v2v":
        v2v_users = [u for u in users if isinstance(u, User)]
        user_dict = {u.user_id: u for u in v2v_users}
        for sender in v2v_users:
            for receiver in v2v_users:
                if sender.user_id != receiver.user_id:
                    sender.protocol = protocole
                    receiver.protocol = protocole
                    sender.send_message(receiver, size=1)
            sender.process_queue(user_dict, metric)

    elif mode == "v2i":
        infra_nodes = [u for u in users if isinstance(u, Infrastructure)]
        normal_users = [u for u in users if isinstance(u, User)]
        if len(normal_users) <= 1 or not infra_nodes:
            return metric
        for u in normal_users:
            for v in normal_users:
                if u.user_id != v.user_id:
                    for infra in infra_nodes:
                        u.protocol = protocole
                        infra.protocol = protocole
                        v.protocol = protocole
                        u.send_message(infra, size=1)
                        infra.process_queue({infra.user_id: infra}, metric)
                        infra.send_message(v, size=1)
                        infra.process_queue({v.user_id: v}, metric)
    return metric

def main():
    print("[Étape] Démarrage de la simulation")
    failed_times: List[float] = []
    users = charger_usagers_depuis_csv("sumoTrace_edge.csv")
    print(f"[Étape] {len(users)} usagers chargés")
    groups = regrouper_par_temps(users)
    print(f"[Étape] {len(groups)} groupes temporels créés")

    protocole_v2v = Protocole("V2V", network_load=0.1, packet_loss_rate=0.1, transmission_time=0.1)
    protocole_v2i = Protocole("V2I", network_load=0.1, packet_loss_rate=0.05, transmission_time=0.5)

    with open('resultats_v2v.csv', 'w', newline='') as csvfile:
        print("[Étape] Écriture des résultats dans 'resultats_v2v.csv'")
        writer = csv.writer(csvfile)
        writer.writerow(["Temps", "Protocole", "Délai moyen (s)", "Taux de perte (%)", "Charge moyenne"])
        for t in sorted(groups):
            batch = groups[t]
            if len(batch) <= 1:
                failed_times.append(t)
                continue

            mv2v = simuler_communication(batch, protocole_v2v, Metric(), "v2v")
            avg, loss, load = mv2v.get_metrics()

            # Si la simulation échoue, on ajoute le temps à failed_times et on continue à la prochaine itération
            if avg is None:
                failed_times.append(t)
                continue

            # Écriture des résultats : toujours deux lignes, "N/A" pour protocole en échec
            # Ligne V2V
            writer.writerow([
                t,
                protocole_v2v.name,
                round(avg, 4) if avg is not None else "N/A",
                round(loss * 100, 2) if loss is not None else "N/A",
                round(load, 4) if load is not None else "N/A",
            ])

    with open('resultats_V2I.csv', 'w', newline='') as csvfile:
                
        print("[Étape] Écriture des résultats dans 'resultats_V2I.csv'")
        writer = csv.writer(csvfile)
        writer.writerow(["Temps", "Protocole", "Délai moyen (s)", "Taux de perte (%)", "Charge moyenne"])
        for t in sorted(groups):
            batch = groups[t]
            if len(batch) <= 1:
                failed_times.append(t)
                continue

            # Simulation V2I
            mv2i = simuler_communication(batch, protocole_v2i, Metric(), "v2i")
            avg, loss, load = mv2i.get_metrics()

            # Si la simulation échoue, on ajoute le temps à failed_times et on continue à la prochaine itération
            if avg is None:
                failed_times.append(t)
                continue
            
            # Écriture des résultats
            writer.writerow([
                t,
                protocole_v2i.name,
                round(avg, 4) if avg is not None else "N/A",
                round(loss * 100, 2) if loss is not None else "N/A",
                round(load, 4) if load is not None else "N/A",
            ])
    print(f"[Résultat] Temps sans communications réussies : {failed_times}")
