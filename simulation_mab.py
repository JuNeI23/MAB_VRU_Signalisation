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
        self.taille = size
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
        distance = self.distance_to(receiver)
        if distance <= self.range and self.usager_id != receiver.usager_id and size <= self.processing_capacity:
            message = Message(self.usager_id, receiver.usager_id, size)
            overall_priority = self.priority + message.priority
            self.queue.put((overall_priority, message))
        else:
            print(f"Message not sent. Distance: {distance}, Range: {self.range}, Size: {size}, Processing Capacity: {self.processing_capacity}")

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
                metrics.update_metrics(transmission_delay, queue_delay, message.size, self.protocol.network_load)       

    def mettre_a_jour_position(self, x: float, y: float, speed: float, angle: float, time: float):
        self.x = x
        self.y = y
        self.speed = speed
        self.angle = angle
        self.time = time

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
    def __init__(self, protocole):
        self.protocole = protocole
        self.messages_envoyes = 0
        self.messages_recus = 0
        self.messages_perdus = 0
        self.charge_totale = 0
        self.delai = 0

    def update_metrics(self, succes):
        self.messages_envoyes += 1
        if succes:
            self.messages_recus += 1
            self.charge_totale += self.protocole.charge_par_message
            # Simuler le délai de transmission
            # En ajoutant le temps de transmission du protocole
            # au délai total
            self.delai += self.protocole.temps_transmission
        else:
            self.messages_perdus += 1
            
    def get_metrics(self):
        taux_perte = self.messages_perdus / self.messages_envoyes if self.messages_envoyes > 0 else 0
        delai_moyen = self.delai / self.messages_recus if self.messages_recus > 0 else 0
        charge_moyenne = self.charge_totale / self.messages_envoyes if self.messages_envoyes > 0 else 0
        return  self.delai, delai_moyen, taux_perte, charge_moyenne
        
# Classe pour représenter un protocole de communication
# entre les utilisateurs
# Chaque protocole a un nom, un temps de transmission et un taux de réussite
# La classe a une méthode pour transmettre un message entre deux utilisateurs
# selon le protocole
class Protocole:
    def __init__(self, nom, temps_transmission, taux_reussite, charge_par_message):
        self.nom = nom
        self.temps_transmission = temps_transmission  # en secondes
        self.taux_reussite = taux_reussite  # probabilité de succès (entre 0 et 1)
        self.charge_par_message = charge_par_message  # en Ko

    def transmettre(self, message, emetteur, recepteur):
        """
        Simule l'envoi d'un message entre deux utilisateurs selon le protocole.
        Retourne True si la transmission réussit, False sinon.
        """
        time.sleep(self.temps_transmission)  # simule le délai
        if random.random() < self.taux_reussite:
            return True
        return False


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

if __name__ == "__main__":  
    # Définir les protocoles
    protocole_v2v = Protocole("V2V", temps_transmission=0.1, taux_reussite=0.9, charge_par_message=1)
    protocole_v2i = Protocole("V2I", temps_transmission=0.5, taux_reussite=0.95, charge_par_message=2)

    # Lire les usagers depuis un fichier CSV
    usagers = charger_usagers_depuis_csv("sumoTrace.csv")

    # Simuler la communication V2V
    metric_v2v = Metric(protocole_v2v)
    metric_v2v = simuler_communication(usagers, protocole_v2v, metric_v2v)

    # Simuler la communication V2I
    metric_v2i = Metric(protocole_v2i)
    metric_v2i = simuler_communication(usagers, protocole_v2i, metric_v2i)


    # Afficher les résultats
    print(f"Protocole V2V: {metric_v2v}")
    print(f"Protocole V2I: {metric_v2i}")
    print("Métriques V2V:", metric_v2v.get_metrics())
    print("Métriques V2I:", metric_v2i.get_metrics())

    # Sauvegarder les résultats dans un fichier CSV
    with open('resultats.csv', mode='w', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
        writer.writerow(["Protocole", "Messages envoyés", "Messages reçus", "Messages perdus", "Charge totale", "Délai total"])
        writer.writerow([protocole_v2v.nom, metric_v2v.messages_envoyes, metric_v2v.messages_recus,
                         metric_v2v.messages_perdus, metric_v2v.charge_totale, metric_v2v.delai])
        writer.writerow([protocole_v2i.nom, metric_v2i.messages_envoyes, metric_v2i.messages_recus,
                         metric_v2i.messages_perdus, metric_v2i.charge_totale, metric_v2i.delai])
    
 