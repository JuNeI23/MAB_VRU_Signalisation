import random
import time

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

    def enregistrer_envoi(self, succes, taille=0):
        self.messages_envoyes += 1
        if succes:
            self.messages_recus += 1
            self.charge_totale += taille
            self.delai += self.protocole.temps_transmission
        else:
            self.messages_perdus += 1

    def taux_perte(self):
        if self.messages_envoyes == 0:
            return 0.0
        return self.messages_perdus / self.messages_envoyes

    def charge_reseau(self):
        return self.messages_envoyes

    def delai_moyen(self):
        if self.messages_recus == 0:
            return 0.0
        return self.delai / self.messages_recus

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
