import logging
from typing import List, Dict
import pandas as pd
import csv

from models import User, Infrastructure, Node
from protocols import Protocole, Metric

logger = logging.getLogger(__name__)

def charger_usagers_depuis_csv(fichier_csv: str) -> List[Node]:
    logger.info("Loading users from CSV: %s", fichier_csv)
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
    logger.info("%d users loaded", len(usagers))
    return usagers

def regrouper_par_temps(usagers: List[Node]) -> Dict[float, List[Node]]:
    logger.info("Grouping users by time")
    result: Dict[float, List[Node]] = {}
    for u in usagers:
        result.setdefault(u.time, []).append(u)
    return result

def simuler_communication(users: List[Node], protocole: Protocole, metric: Metric, mode: str = "v2v") -> Metric:
    if mode == "v2v":
        logger.info("Running V2V simulation with %s", protocole.name)
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
        logger.info("Running V2I simulation with %s", protocole.name)
        infra_nodes = [u for u in users if isinstance(u, Infrastructure)]
        normal_users = [u for u in users if isinstance(u, User)]
        if len(normal_users) <= 1 or not infra_nodes:
            logger.warning("No possible V2I exchanges")
            return metric
        for u in normal_users:
            for v in normal_users:
                if u.user_id != v.user_id:
                    for infra in infra_nodes:
                        u.protocol = protocole
                        infra.protocol = protocole
                        v.protocol = protocole
                        logger.debug("V2I %s -> infra %s", u.user_id, infra.user_id)
                        u.send_message(infra, size=1)
                        infra.process_queue({infra.user_id: infra}, metric)
                        logger.debug("V2I infra %s -> %s", infra.user_id, v.user_id)
                        infra.send_message(v, size=1)
                        infra.process_queue({v.user_id: v}, metric)
    return metric

def main():
    logging.basicConfig(level=logging.INFO)
    users = charger_usagers_depuis_csv("sumoTrace_edge.csv")
    groups = regrouper_par_temps(users)

    protocole_v2v = Protocole("V2V", network_load=0.1, packet_loss_rate=0.1, transmission_time=0.1)
    protocole_v2i = Protocole("V2I", network_load=0.1, packet_loss_rate=0.05, transmission_time=0.5)

    with open('resultats.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Temps", "Protocole", "Délai moyen (s)", "Taux de perte (%)", "Charge moyenne"])
        for t in sorted(groups):
            batch = groups[t]
            if len(batch) <= 1:
                writer.writerow([t, protocole_v2v.name, "N/A", "N/A", "N/A"])
                writer.writerow([t, protocole_v2i.name, "N/A", "N/A", "N/A"])
                continue

            mv2v = simuler_communication(batch, protocole_v2v, Metric(), "v2v")
            avg, loss, load = mv2v.get_metrics()
            writer.writerow([
                t,
                protocole_v2v.name,
                round(avg, 4) if avg is not None else "Ø",
                round(loss * 100, 2) if loss is not None else "Ø",
                round(load, 4) if load is not None else "Ø",
            ])

            mv2i = simuler_communication(batch, protocole_v2i, Metric(), "v2i")
            avg2, loss2, load2 = mv2i.get_metrics()
            writer.writerow([
                t,
                protocole_v2i.name,
                round(avg2, 4) if avg2 is not None else "Ø",
                round(loss2 * 100, 2) if loss2 is not None else "Ø",
                round(load2, 4) if load2 is not None else "Ø",
            ])

    logger.info("Simulation complete. Results written to 'resultats.csv'.")