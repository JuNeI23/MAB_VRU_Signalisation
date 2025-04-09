import random
import time
import csv

DISTANCE_MAX = 50

class Message:
    def __init__(self, sender_id, receiver_id, content, taille=1):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.content = content
        self.taille = taille
        self.timestamp = time.time()

class User:
    def __init__(self, user_id, user_type='pieton', position=0):
        self.user_id = user_id
        self.user_type = user_type
        self.inbox = []
        self.position = position
        priorite_map = {'pieton': 2, 'vehicule': 1, 'infrastructure': 0}
        self.priorite = priorite_map.get(user_type, 0)

    def send_message(self, receiver, content, taille=1):
        distance = abs(self.position - receiver.position)
        if distance <= DISTANCE_MAX:
            msg = Message(self.user_id, receiver.user_id, content, taille)
            receiver.receive_message(msg)
            return msg
        return None

    def receive_message(self, message):
        self.inbox.append(message)

    def move(self, deplacement):
        self.position += deplacement

class Infrastructure(User):
    def __init__(self, user_id, position=0):
        super().__init__(user_id, user_type='infrastructure', position=position)

class Metric:
    def __init__(self, protocole):
        self.protocole = protocole
        self.messages_envoyes = 0
        self.messages_recus = 0
        self.messages_perdus = 0
        self.charge_totale = 0
        self.delai = 0

    def update_metrics(self, succes, taille=0):
        self.messages_envoyes += 1
        if succes:
            self.messages_recus += 1
            self.charge_totale += taille
            self.delai += self.protocole.temps_transmission
        else:
            self.messages_perdus += 1
            
class Protocole:
    def __init__(self, nom, temps_transmission, taux_reussite, charge_par_message):
        self.nom = nom
        self.temps_transmission = temps_transmission  # en secondes
        self.taux_reussite = taux_reussite  # probabilité de succès (entre 0 et 1)
        self.charge_par_message = charge_par_message  # charge moyenne ajoutée par message

    def transmettre(self, message, emetteur, recepteur):
        """
        Simule l'envoi d'un message entre deux utilisateurs selon le protocole.
        Retourne True si la transmission réussit, False sinon.
        """
        time.sleep(self.temps_transmission)  # simule le délai
        if random.random() < self.taux_reussite:
            recepteur.receive_message(message)
            return True
        return False

def lire_usagers_depuis_csv(fichier_csv):
    usagers = []
    with open(fichier_csv, newline='') as csvfile:
        lecteur = csv.DictReader(csvfile)
        for ligne in lecteur:
            user_id = ligne['user_id']
            user_type = ligne.get('user_type', 'pieton')
            position = int(ligne.get('position', 0))

            if user_type == 'infrastructure':
                usager = Infrastructure(user_id, position)
            else:
                usager = User(user_id, user_type, position)

            usagers.append(usager)
    return usagers

def simuler_communication(usagers, protocole, distance_max=DISTANCE_MAX):
    """
    Simule la communication entre les utilisateurs selon le protocole donné.
    """
    for emetteur in usagers:
        for recepteur in usagers:
            if emetteur != recepteur:
                distance = abs(emetteur.position - recepteur.position)
                if distance <= distance_max:
                    message = emetteur.send_message(recepteur, "Hello", taille=1)
                    if message:
                        protocole.transmettre(message, emetteur, recepteur)
                        protocole.enregistrer_envoi(True, taille=1)
                    else:
                        protocole.enregistrer_envoi(False)
    return protocole

def main():    
    # Définir les protocoles
    protocole_v2v = Protocole("V2V", temps_transmission=0.1, taux_reussite=0.9, charge_par_message=1)
    protocole_v2i = Protocole("V2I", temps_transmission=0.5, taux_reussite=0.95, charge_par_message=2)

    # Lire les usagers depuis un fichier CSV
    usagers = lire_usagers_depuis_csv('usagers.csv')

    # Simuler la communication V2V
    metric_v2v = Metric(protocole_v2v)
    metric_v2v = simuler_communication(usagers, protocole_v2v)

    # Simuler la communication V2I
    metric_v2i = Metric(protocole_v2i)
    metric_v2i = simuler_communication(usagers, protocole_v2i)

    # Afficher les résultats
    print(f"Protocole V2V: {metric_v2v}")
    print(f"Protocole V2I: {metric_v2i}")