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

# Classe pour représenter un utilisateur
# Chaque utilisateur a un identifiant, une position (x, y), un angle, une vitesse, une position sur la route,
# un bord (edge), un temps, un type d'utilisateur (vru ou véhicule) et une catégorie
# (vru ou véhicule)
# La classe a des méthodes pour envoyer et recevoir des messages, mettre à jour la position de l'utilisateur
# et calculer la distance entre deux utilisateurs
class User:
    def __init__(self, usager_id: str, x: float, y: float, angle: float, speed: float, position: float,
                 lane: str, time: float, usager_type: str = "DEFAULT", categorie: str = "vru"):
        self.usager_id = usager_id
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.position = position 
        self.lane = lane
        self.slope = 0
        self.usager_type = usager_type
        self.time = time
        self.categorie = categorie  # "vru" ou "vehicule"
        self.queue = queue.PriorityQueue()
        
        # Initialiser la priorité et la portée en fonction de la catégorie
        if self.categorie == "vru":
            self.priority = 1
            self.range = 90
            self.processing_capacity = 1
            self.queue_size = 10
        elif self.categorie == "vehicule":
            self.priority = 2
            self.range = 120
            self.processing_capacity = 2
            self.queue_size = 50
        else:
            raise ValueError("Invalid category. Must be 'vru' or 'vehicule'.")
    
    def send_message(self, receiver, size):
        if size <= self.processing_capacity and self.within_range(receiver):
            message = Message(self.usager_id, receiver.usager_id, self.priority, size)
            overall_priority = self.priority + message.priority
            self.queue.put((overall_priority, message))
            print(f"Message sent from {self.usager_id} to {receiver.usager_id}. Distance: {self.distance_to(receiver)}, Range: {self.range}, Size: {size}, Processing Capacity: {self.processing_capacity}")
        else:
            if not self.within_range(receiver):
                print(f"{self.usager_id} cannot reach {receiver.usager_id} (distance: {self.distance_to(receiver)}, range: {self.range})")
            else:
                print(f"Message not sent. Distance: {self.distance_to(receiver)}, Range: {self.range}, Size: {size}, Processing Capacity: {self.processing_capacity}")

    def process_queue(self, users, metric):
        messages_per_receiver = {}  # Dictionnaire pour stocker les messages par destinataire

        while not self.queue.empty():
            _, message = self.queue.get()
            receiver = users.get(message.receiver_id, None)
            # Ajout impressions pour comprendre pourquoi les messages ne passent pas
            if receiver:
                if not self.within_range(receiver):
                    print(f"[SKIP] {self.usager_id} → {receiver.usager_id}: hors de portée")
                elif self.protocol.network_load > 0.8:
                    print(f"[SKIP] {self.usager_id} → {receiver.usager_id}: charge réseau trop élevée ({self.protocol.network_load:.2f})")
            # Condition modifiée : désactivation de la charge réseau pour test
            if receiver and self.within_range(receiver):  # condition de charge réseau désactivée pour test
                if receiver not in messages_per_receiver:
                    messages_per_receiver[receiver] = []  # Initialiser une nouvelle file d'attente pour le destinataire
                messages_per_receiver[receiver].append(message)  # Ajouter le message à la file d'attente du destinataire

        for receiver, messages in messages_per_receiver.items():
            messages.sort(key=lambda msg: msg.priority)  # Trier les messages par priorité
            for message in messages:
                transmission_delay = self.protocol.transmit_message(self, message, receiver)
                queue_delay = time.time() - message.timestamp
                print(f"[DEBUG] Update metric for message from {self.usager_id} to {receiver.usager_id}")
                metric.update_metrics(transmission_delay, queue_delay, self.protocol.network_load)
    
    def within_range(self, other):
        # Vérifier si l'autre utilisateur est dans la portée de l'utilisateur
        return self.distance_to(other) <= self.range
    
    # Méthode pour calculer la distance entre deux utilisateurs
    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def receive_message(self, message):
        return time.time() - message.timestamp

# Classe pour représenter les métriques de communication
# entre les utilisateurs
# La classe a des attributs pour le protocole utilisé, le nombre de messages envoyés,
# le nombre de messages reçus, le nombre de messages perdus, la charge totale et le délai total
# La classe a des méthodes pour mettre à jour les métriques, obtenir les métriques et afficher les résultats
class Metric:
    def __init__(self):
        self.total_transmission_delay = 0
        self.total_reception_delay = 0
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
            receiver.receive_message(message)
            self.update_network_load()
            return delay
        else:
            return None
    
    def update_network_load(self):
        self.network_load = random.random()  # Update network load based on random value for demonstration purposes
"""
class Infrastructure:
    def __init__(self, id, protocol, processing_capacity):
        self.id = id
        self.protocol = protocol
        self.processing_capacity = processing_capacity
        self.user_type = 'Infrastructure'
        self.queue = queue.PriorityQueue()

    def send_message(self, receiver, size):
        if size <= self.processing_capacity:
            message = Message(self.id, receiver.usager_id, self.priority, size)
            overall_priority = self.priority + message.priority
            self.queue.put((overall_priority, message))
        else:
            print(f"Message not sent. Distance: {self.distance_to(receiver)}, Range: {self.range}, Size: {size}, Processing Capacity: {self.processing_capacity}")
    
    def receive_message(self, message):
        return time.time() - message.timestamp

    def process_queue(self, users, metric):
        transmission_delay = None
        queue_delay = None
        messages_per_receiver = {}  # Dictionnaire pour stocker les messages par destinataire

        while not self.queue.empty():
            _, message = self.queue.get()
            receiver = users.get(message.receiver_id, None)
            if receiver:
                if receiver not in messages_per_receiver:
                    messages_per_receiver[receiver] = []  # Initialiser une nouvelle file d'attente pour le destinataire
                messages_per_receiver[receiver].append(message)  # Ajouter le message à la file d'attente du destinataire
    
        for receiver, messages in messages_per_receiver.items():
            messages.sort(key=lambda msg: msg.priority)  # Trier les messages par priorité
            for message in messages:
                transmission_delay = self.transmit_message(message, receiver)
                queue_delay = time.time() - message.timestamp
                metric.update_metrics(transmission_delay, queue_delay, message.size, self.protocol.network_load)
    

    def move(self):
        pass

    def within_range(self, user):
        distance = abs(self.id - user.id)
        return distance <= self.range
"""

# Fonction pour charger les usagers depuis un fichier CSV
# La fonction prend en entrée le nom du fichier CSV et retourne une liste d'utilisateurs
# Chaque ligne du fichier CSV représente un usager avec ses attributs
def charger_usagers_depuis_csv(fichier_csv: str) -> List[User]:
    print("Chargement des usagers depuis le fichier CSV...")
    df = pd.read_csv(fichier_csv)
    usagers = []

    for _, ligne in df.iterrows():
        time = ligne['_time']

        # Ajouter les piétons (VRUs)
        if pd.notna(ligne['person/_id']):
            usager = User(
                usager_id=ligne['person/_id'],
                x=ligne['person/_x'],
                y=ligne['person/_y'],
                angle=ligne['person/_angle'],
                speed=ligne['person/_speed'],
                position=ligne['person/_pos'],
                lane=ligne['person/_edge'],
                time=time,
                usager_type=ligne['person/_type'],
                categorie="vru"
            )
            usagers.append(usager)

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
def extraire_usagers_par_temps(usagers: List[User]) -> dict:
    print("Extraction des usagers par temps...")
    usagers_par_temps = {}
    for usager in usagers:
        temps = usager.time
        if temps not in usagers_par_temps:
            usagers_par_temps[temps] = []
        usagers_par_temps[temps].append(usager)
    print(f"Extraction terminée : {len(usagers_par_temps)} timestamps trouvés.")
    return usagers_par_temps

def simuler_communication(users: List[User], protocole: Protocole, metric: Metric):
    print(f"Simulation de communication avec le protocole {protocole.name}")
    # Simuler l'envoi et la réception de messages entre tous les usagers
    for i, emetteur in enumerate(users):
        for j, recepteur in enumerate(users):
            if i != j:
                emetteur.protocol = protocole  # affecter le protocole
                recepteur.protocol = protocole  # affecter le protocole aussi au récepteur
                try:
                    emetteur.send_message(recepteur, size=1)
                    print(f"Message envoyé de {emetteur.usager_id} à {recepteur.usager_id}")
                except:
                    continue
        emetteur.protocol = protocole
        emetteur.process_queue({u.usager_id: u for u in users}, metric)
    return metric

if __name__ == "__main__":
    print("Début de la simulation...")
    usagers = charger_usagers_depuis_csv("sumoTrace.csv")
    usagers_par_temps = extraire_usagers_par_temps(usagers)

    protocole_v2v = Protocole("V2V", network_load=0.1, packet_loss_rate=0.1, transmission_time=0.1)
    protocole_v2i = Protocole("V2I", network_load=0.1, packet_loss_rate=0.05, transmission_time=0.5)

    with open('resultats.csv', mode='w', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
        writer.writerow(["Temps", "Protocole", "Délai moyen (s)", "Taux de perte (%)", "Charge moyenne"])

        for temps, usagers_t in sorted(usagers_par_temps.items()):
            if len(usagers_t) <= 1:
                print(f"Aucun échange possible à t={temps} (1 seul usager)")
                writer.writerow([temps, protocole_v2v.name, "N/A", "N/A", "N/A"])
                continue
            # V2V
            metric_v2v = Metric()
            metric_v2v = simuler_communication(usagers_t, protocole_v2v, metric_v2v)
            delai_moyen, taux_perte, charge_moyenne = metric_v2v.get_metrics()
            print(metric_v2v.total_messages, metric_v2v.lost_messages)
            writer.writerow([
                temps, protocole_v2v.name,
                round(delai_moyen, 4) if delai_moyen is not None else "Ø",
                round(taux_perte * 100, 2) if taux_perte is not None else "Ø",
                round(charge_moyenne, 4) if charge_moyenne is not None else "Ø"
            ])
            print(f"Temps {temps} - Protocole {protocole_v2v.name} terminé.")

            """
            # V2I
            metric_v2i = Metric()
            metric_v2i = simuler_communication(usagers_t, protocole_v2i, metric_v2i)
            delai_moyen, taux_perte, charge_moyenne = metric_v2i.get_metrics()
            writer.writerow([
                temps, protocole_v2i.name,
                round(delai_moyen, 4) if delai_moyen is not None else "N/A",
                round(taux_perte * 100, 2) if taux_perte is not None else "N/A",
                round(charge_moyenne, 4) if charge_moyenne is not None else "N/A"
            ])
            
            print(f"Temps {temps} - Protocole {protocole_v2i.name} terminé.")
            """
    print("Simulation terminée. Résultats écrits dans 'resultats.csv'.")