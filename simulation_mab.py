import random
import time
import queue
import csv
from typing import List

import pandas as pd

# Classe pour représenter un message
# entre deux utilisateurs
# Chaque message a un identifiant d'émetteur, un identifiant de récepteur, une taille et un timestamp
# pour simuler le temps d'envoi
# et de réception du message
class Message:
    def __init__(self, sender_id, receiver_id, priority, size):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.priority = priority
        self.size = size
        self.timestamp = time.time()

    def __lt__(self, other):
        return self.priority < other.priority

# Classe de base Node
class Node:
    def __init__(self, x: float, y: float, range_: float, priority: int, processing_capacity: int, user_id: str = None):
        self.x = x
        self.y = y
        self.range = range_
        self.priority = priority
        self.processing_capacity = processing_capacity
        self.queue = queue.PriorityQueue()
        self.user_id = user_id

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def within_range(self, other):
        return self.distance_to(other) <= self.range

    def send_message(self, receiver, size):
        distance = self.distance_to(receiver)
        if size <= self.processing_capacity and self.within_range(receiver):
            receiver_id = getattr(receiver, 'user_id', getattr(receiver, 'id', None))
            message = Message(self.user_id, receiver_id, self.priority, size)
            self.queue.put((message.priority, message))
            print(f"[INFO] {self.user_id} → {receiver_id} : message prêt (distance: {distance:.2f})")
        else:
            print(f"[ERROR] {self.user_id} → {getattr(receiver, 'user_id', getattr(receiver, 'id', None))} : size too big or out of range (distance: {distance:.2f}, range: {self.range})")

    def process_queue(self, users, metric):
        messages_per_receiver = {}
        while not self.queue.empty():
            _, message = self.queue.get()
            receiver = users.get(message.receiver_id, None)
            if receiver:
                if receiver not in messages_per_receiver:
                    messages_per_receiver[receiver] = []
                messages_per_receiver[receiver].append(message)
        for receiver, messages in messages_per_receiver.items():
            messages.sort(key=lambda msg: msg.priority)
            for message in messages:
                if not getattr(receiver, 'protocol', None):
                    receiver.protocol = self.protocol
                transmission_delay = self.protocol.transmit_message(self, message, receiver)
                queue_delay = time.time() - message.timestamp
                metric.update_metrics(transmission_delay, queue_delay, self.protocol.network_load)

# Classe User héritant de Node
class User(Node):
    def __init__(self, usager_id: str, x: float, y: float, angle: float, speed: float, position: float,
                 lane: str, time: float, usager_type: str = "DEFAULT", categorie: str = "vru"):
        if categorie == "vru":
            priority = 1
            range_ = 90
            processing_capacity = 1
        elif categorie == "vehicule":
            priority = 2
            range_ = 120
            processing_capacity = 2
        else:
            raise ValueError("Invalid category. Must be 'vru' or 'vehicule'.")
        super().__init__(x, y, range_, priority, processing_capacity, user_id=usager_id)
        self.angle = angle
        self.speed = speed
        self.position = position
        self.lane = lane
        self.usager_type = usager_type
        self.time = time
        self.categorie = categorie
        self.protocol = None
        self.queue_size = 10 if categorie == "vru" else 50

# Classe pour représenter les métriques de communication
# entre les utilisateurs
# La classe a des attributs pour le protocole utilisé, le nombre de messages envoyés,
# le nombre de messages reçus, le nombre de messages perdus, la charge totale et le délai total
# La classe a des méthodes pour mettre à jour les métriques, obtenir les métriques et afficher les résultats
class Metric:
    def __init__(self):
        self.total_transmission_delay = 0
        self.total_queue_delay = 0  # New variable for queue delay
        self.total_messages = 0
        self.lost_messages = 0
        self.total_network_load = 0

    def update_metrics(self, transmission_delay, queue_delay, network_load):
        if transmission_delay:
            self.total_transmission_delay += transmission_delay
            self.total_queue_delay += queue_delay
            self.total_messages += 1
            self.total_network_load += network_load
        else:
            self.lost_messages += 1
            
    def get_metrics(self):
        total_delay = self.total_transmission_delay + self.total_queue_delay  # Calculate total delay
        average_delay = total_delay / self.total_messages if self.total_messages != 0 else None
        packet_loss_rate = self.lost_messages / (self.total_messages + self.lost_messages) if self.total_messages + self.lost_messages != 0 else None
        average_network_load = self.total_network_load / self.total_messages if self.total_messages != 0 else None
        return average_delay, packet_loss_rate, average_network_load
        

class Protocole:
    def __init__(self, name, network_load , packet_loss_rate, transmission_time, transmission_success_rate=0.9):
        self.name = name
        self.network_load = network_load
        self.packet_loss_rate = packet_loss_rate  # probabilité de perte de paquet (entre 0 et 1)
        self.transmission_time = transmission_time  # en secondes
        self.transmission_success_rate = transmission_success_rate  # probabilité de succès (entre 0 et 1)

    def transmit_message(self, sender, message, receiver):
        if receiver and random.random() < self.transmission_success_rate:  # Simulate packet loss
            # Calculate distance
            distance = sender.distance_to(receiver)
            if distance > sender.range:
                print(f"Message not sent. Distance: {distance}, Range: {sender.range}")
                return None
            # Add delay proportional to distance
            delay = self.transmission_time + 0.01 * distance  # Adjust the constant factor as needed
            time.sleep(delay)
            self.update_network_load()
            return delay
        else:
            return None
    
    def update_network_load(self):
        self.network_load = random.random()  # Update network load based on random value for demonstration purposes

# Classe Infrastructure héritant de Node
class Infrastructure(Node):
    def __init__(self, id, protocol, x, y, processing_capacity, time):
        super().__init__(x, y, range_=300, priority=1, processing_capacity=processing_capacity, user_id=id)
        self.id = id
        self.protocol = protocol
        self.time = time
        self.user_type = 'Infrastructure'


# Fonction pour charger les usagers depuis un fichier CSV
# La fonction prend en entrée le nom du fichier CSV et retourne une liste d'utilisateurs
# Chaque ligne du fichier CSV représente un usager avec ses attributs
def charger_usagers_depuis_csv(fichier_csv: str) -> List[User]:
    print("Chargement des usagers depuis le fichier CSV...")
    df = pd.read_csv(fichier_csv)
    usagers = []

    for _, ligne in df.iterrows():
        time = ligne['_time']

        # Ajouter les piétons et l'infrastructure
        if pd.notna(ligne['person/1/_id']):
            # Création de l'infrastructure à partir de person/0
            protocole_infra = Protocole("V2I", network_load=0.1, packet_loss_rate=0.05, transmission_time=0.5)
            infra = Infrastructure(
                id=ligne['person/1/_id'],
                protocol=protocole_infra,
                x=ligne['person/1/_x'],
                y=ligne['person/1/_y'],
                time=time,
                processing_capacity=100
            )
            usagers.append(infra)

        if pd.notna(ligne['person/0/_id']):
            # Création d'un utilisateur normal à partir de person/1
            user = User(
                usager_id=ligne['person/0/_id'],
                x=ligne['person/0/_x'],
                y=ligne['person/0/_y'],
                angle=ligne['person/0/_angle'],
                speed=ligne['person/0/_speed'],
                position=ligne['person/0/_pos'],
                lane=ligne['person/0/_edge'],
                time=ligne['_time'],
                usager_type=ligne['person/0/_type'],
                categorie="vru"
            )
            usagers.append(user)

        # Ajouter les véhicules présents à cette ligne
        i = 0
        while f"vehicle/{i}/_id" in ligne:
            veh_id = ligne[f"vehicle/{i}/_id"]
            if pd.notna(veh_id):
                usager = User(
                    usager_id=veh_id,
                    x=ligne[f"vehicle/{i}/_x"],
                    y=ligne[f"vehicle/{i}/_y"],
                    angle=ligne[f"vehicle/{i}/_angle"],
                    speed=ligne[f"vehicle/{i}/_speed"],
                    lane=ligne[f"vehicle/{i}/_lane"],
                    position=ligne[f"vehicle/{i}/_pos"],
                    time=time,
                    usager_type=ligne[f"vehicle/{i}/_type"],
                    categorie="vehicule"
                )
                usagers.append(usager)
            i += 1

    print(f"{len(usagers)} usagers chargés.")
    return usagers

# Fonctions additionnelles demandées
# Renommé extraire_usagers_par_temps → regrouper_par_temps
def regrouper_par_temps(usagers: List[User]) -> dict:
    print("Extraction des usagers par temps (y compris infrastructures)...")
    usagers_par_temps = {}
    for usager in usagers:
        temps = getattr(usager, 'time', 0)
        if temps not in usagers_par_temps:
            usagers_par_temps[temps] = []
        usagers_par_temps[temps].append(usager)
        print(f"{type(usager).__name__} {getattr(usager, 'user_id', getattr(usager, 'id', 'unknown'))} ajouté à t={temps}")
    print(f"Extraction terminée : {len(usagers_par_temps)} timestamps trouvés.")
    return usagers_par_temps

# Fonction pour simuler la communication entre les utilisateurs
def simuler_communication(users: List[User], protocole: Protocole, metric: Metric, mode: str = "v2v") -> Metric:
    if mode == "v2v":
        print(f"Simulation de communication V2V avec le protocole {protocole.name}")
        v2v_users = [u for u in users if isinstance(u, User)]
        for i, emetteur in enumerate(v2v_users):
            for j, recepteur in enumerate(v2v_users):
                if i != j:
                    emetteur.protocol = protocole
                    recepteur.protocol = protocole
                    try:
                        emetteur.send_message(recepteur, size=1)
                        print(f"Message envoyé de {emetteur.user_id} à {recepteur.user_id}")
                    except Exception as e:
                        print(f"[ERROR] {e}")
            emetteur.process_queue({u.user_id: u for u in v2v_users}, metric)

    elif mode == "v2i":
        print(f"Simulation de communication V2I avec le protocole {protocole.name}")
        infrastructures = [u for u in users if isinstance(u, Infrastructure)]
        usagers_normaux = [u for u in users if isinstance(u, User)]
        # Ajout de la condition demandée pour éviter les échanges V2I impossibles
        if len(usagers_normaux) <= 1 or len(infrastructures) == 0:
            print(f"Aucun échange V2I possible (usagers: {len(usagers_normaux)}, infrastructures: {len(infrastructures)})")
            return metric
        for emetteur in usagers_normaux:
            for recepteur in usagers_normaux:
                if emetteur.user_id != recepteur.user_id:
                    for infra in infrastructures:
                        try:
                            emetteur.protocol = protocole
                            infra.protocol = protocole
                            recepteur.protocol = protocole
                            print(f"[V2I] {emetteur.user_id} → infra {infra.id}")
                            emetteur.send_message(infra, size=1)
                            infra.process_queue({getattr(infra, 'user_id', getattr(infra, 'id', None)): infra}, metric)
                            print(f"[V2I] infra {infra.id} → {recepteur.user_id}")
                            infra.send_message(recepteur, size=1)
                            infra.process_queue({getattr(recepteur, 'user_id', getattr(recepteur, 'id', None)): recepteur}, metric)
                        except Exception as e:
                            print(f"Erreur V2I : {e}")
    return metric


if __name__ == "__main__":
    print("Début de la simulation...")
    usagers = charger_usagers_depuis_csv("sumoTrace_edge.csv")
    usagers_par_temps = regrouper_par_temps(usagers)

    protocole_v2v = Protocole("V2V", network_load=0.1, packet_loss_rate=0.1, transmission_time=0.1)
    protocole_v2i = Protocole("V2I", network_load=0.1, packet_loss_rate=0.05, transmission_time=0.5)

    with open('resultats.csv', mode='w', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
        writer.writerow(["Temps", "Protocole", "Délai moyen (s)", "Taux de perte (%)", "Charge moyenne"])

        for temps, usagers_t in sorted(usagers_par_temps.items()):
            if len(usagers_t) <= 1:
                print(f"Aucun échange possible à t={temps} (1 seul usager)")
                writer.writerow([temps, protocole_v2v.name, "N/A", "N/A", "N/A"])
                writer.writerow([temps, protocole_v2i.name, "N/A", "N/A", "N/A"])
                continue

            # V2V
            metric_v2v = simuler_communication(usagers_t, protocole_v2v, Metric(), mode="v2v")
            delai_moyen, taux_perte, charge_moyenne = metric_v2v.get_metrics()
            print(metric_v2v.total_messages, metric_v2v.lost_messages)
            writer.writerow([
                temps, protocole_v2v.name,
                round(delai_moyen, 4) if delai_moyen is not None else "Ø",
                round(taux_perte * 100, 2) if taux_perte is not None else "Ø",
                round(charge_moyenne, 4) if charge_moyenne is not None else "Ø"
            ])
            print(f"Temps {temps} - Protocole {protocole_v2v.name} terminé.")

            # V2I
            metric_v2i = simuler_communication(usagers_t, protocole_v2i, Metric(), mode="v2i")
            delai_moyen, taux_perte, charge_moyenne = metric_v2i.get_metrics()
            writer.writerow([
                temps, protocole_v2i.name,
                round(delai_moyen, 4) if delai_moyen is not None else "Ø",
                round(taux_perte * 100, 2) if taux_perte is not None else "Ø",
                round(charge_moyenne, 4) if charge_moyenne is not None else "Ø"
            ])
            print(f"Temps {temps} - Protocole {protocole_v2i.name} terminé.")

    print("Simulation terminée. Résultats écrits dans 'resultats.csv'.")