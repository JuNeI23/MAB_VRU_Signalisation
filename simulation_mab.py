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
                 edge: str, time: float, usager_type: str = "DEFAULT", categorie: str = "vru"):
        self.usager_id = usager_id
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.position = position 
        self.edge = edge
        self.slope = 0
        self.usager_type = usager_type
        self.time = time
        self.categorie = categorie  # "vru" ou "vehicule"
        self.queue = queue.PriorityQueue()
        
        # Initialiser la priorité et la portée en fonction de la catégorie
        if self.categorie == "vru":
            self.priority = 1
            self.range = 10
            self.processing_capacity = 1
            self.queue_size = 10
        elif self.categorie == "vehicule":
            self.priority = 2
            self.range = 20
            self.processing_capacity = 2
            self.queue_size = 50
        else:
            raise ValueError("Invalid category. Must be 'vru' or 'vehicule'.")
    
    def send_message(self, receiver, size=1):
        if self.within_range(receiver) and self.usager_id != receiver.usager_id and size <= self.processing_capacity:
            message = Message(self.usager_id, receiver.usager_id, size)
            overall_priority = self.priority + message.priority
            self.queue.put((overall_priority, message))
        else:
            print(f"Message not sent. Distance: {self.distance_to(receiver)}, Range: {self.range}, Size: {size}, Processing Capacity: {self.processing_capacity}")

    def process_queue(self, users):
        messages_per_receiver = {}  # Dictionnaire pour stocker les messages par destinataire
    
        while not self.queue.empty():
            _, message = self.queue.get()
            receiver = users.get(message.receiver_id, None)
            if receiver and self.within_range(receiver) and not self.protocol.network_load > 0.8:
                if receiver not in messages_per_receiver:
                    messages_per_receiver[receiver] = []  # Initialiser une nouvelle file d'attente pour le destinataire
                messages_per_receiver[receiver].append(message)  # Ajouter le message à la file d'attente du destinataire
    
        for receiver, messages in messages_per_receiver.items():
            messages.sort(key=lambda msg: msg.priority)  # Trier les messages par priorité
            for message in messages:
                transmission_delay = self.protocol.transmit_message(self, message, receiver)
                queue_delay = time.time() - message.timestamp
                Metric.update_metrics(transmission_delay, queue_delay, message.size, self.protocol.network_load)       
    
    def move(self):
        # Mettre à jour la position de l'utilisateur en fonction de sa vitesse et de son angle
        self.x += self.speed * time.cos(self.angle)
        self.y += self.speed * time.sin(self.angle)
        self.position += self.speed
        # Mettre à jour le temps
        self.time += 1

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
        if transmission_delay :
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

class Infrastructure:
    def __init__(self, id, protocol, processing_capacity):
        self.id = id
        self.protocol = protocol
        self.processing_capacity = processing_capacity
        self.user_type = 'Infrastructure'
        self.queue = queue.PriorityQueue()

    def send_message(self, receiver_id, message_priority, size):
        if size <= self.processing_capacity:
            message = Message(self.id, receiver_id, message_priority, size)
            overall_priority =  message_priority
            receiver = users.get(message.receiver_id, None)
            if receiver:
                receiver.queue.put((overall_priority, message))
                print(f'Infrastructure {self.id} has a new message for User {receiver_id} in queue.')
            else:
                print(f'Receiver User {receiver_id} not found.')
        else:
            print(f'Message size {size} exceeds processing capacity of Infrastructure {self.id}.')
    
    def receive_message(self, message):
        return time.time() - message.timestamp

    def process_queue(self, users):
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
                Metric.update_metrics(transmission_delay, queue_delay, message.size, self.protocol.network_load)
    

    def move(self):
        pass

    def within_range(self, user):
        distance = abs(self.id - user.id)
        return distance <= self.range

# Fonction pour charger les usagers depuis un fichier CSV
# La fonction prend en entrée le nom du fichier CSV et retourne une liste d'utilisateurs
# Chaque ligne du fichier CSV représente un usager avec ses attributs


def charger_usagers_depuis_csv(fichier_csv: str) -> List[User]:
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
                edge=ligne['person/_edge'],
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
                    edge=ligne[f"vehicle/{i}/_lane"],
                    time=time,
                    usager_type=ligne[f"vehicle/{i}/_type"],
                    categorie="vehicule"
                )
                usagers.append(usager)
            i += 1

    return usagers

# Fonctions additionnelles demandées
def extraire_usagers_par_temps(usagers: List[User]) -> dict:
    usagers_par_temps = {}
    for usager in usagers:
        temps = usager.time
        if temps not in usagers_par_temps:
            usagers_par_temps[temps] = []
        usagers_par_temps[temps].append(usager)
    return usagers_par_temps

def simuler_communication(users: List[User], protocole: Protocole, metric: Metric):
    # Simuler l'envoi et la réception de messages entre tous les usagers
    for i, emetteur in enumerate(users):
        for j, recepteur in enumerate(users):
            if i != j:
                emetteur.protocol = protocole  # affecter le protocole
                try:
                    emetteur.send_message(recepteur)
                except:
                    continue
        emetteur.protocol = protocole
        emetteur.process_queue({u.usager_id: u for u in users})
    return metric

if __name__ == "__main__":
    # Lire les usagers depuis un fichier CSV
    usagers = charger_usagers_depuis_csv("sumoTrace.csv")
    usagers_par_temps = extraire_usagers_par_temps(usagers)

    # Définir les protocoles
    protocole_v2v = Protocole("V2V", temps_transmission=0.1, taux_reussite=0.9, charge_par_message=1)
    protocole_v2i = Protocole("V2I", temps_transmission=0.5, taux_reussite=0.95, charge_par_message=2)

    # Préparer l'écriture des résultats
    with open('resultats.csv', mode='w', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
        writer.writerow(["Temps", "Protocole", "Messages envoyés", "Messages reçus", "Messages perdus", "Charge totale", "Délai total", "Délai moyen", "Taux de perte", "Charge moyenne"])

        for temps, usagers_t in sorted(usagers_par_temps.items()):
            # V2V
            metric_v2v = Metric(protocole_v2v)
            metric_v2v = simuler_communication(usagers_t, protocole_v2v, metric_v2v)
            delai, delai_moyen, taux_perte, charge_moyenne = metric_v2v.get_metrics()
            writer.writerow([temps, protocole_v2v.nom, metric_v2v.messages_envoyes, metric_v2v.messages_recus, metric_v2v.messages_perdus, metric_v2v.charge_totale, metric_v2v.delai, delai_moyen, taux_perte, charge_moyenne])

            # V2I
            metric_v2i = Metric(protocole_v2i)
            metric_v2i = simuler_communication(usagers_t, protocole_v2i, metric_v2i)
            delai, delai_moyen, taux_perte, charge_moyenne = metric_v2i.get_metrics()
            writer.writerow([temps, protocole_v2i.nom, metric_v2i.messages_envoyes, metric_v2i.messages_recus, metric_v2i.messages_perdus, metric_v2i.charge_totale, metric_v2i.delai, delai_moyen, taux_perte, charge_moyenne])