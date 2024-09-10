import sqlite3
from datetime import datetime

class BDDeManager:
    def __init__(self, db_name='data.db'):
        self.db_name = db_name
        self.conn = None
        self.create_connection()

    def create_connection(self):
        try:
            self.conn = sqlite3.connect(self.db_name)
            print(f"Connected to database: {self.db_name}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")

    def create_table(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    numbers TEXT,
                    date_time TEXT
                )
            ''')
            self.conn.commit()
            print("La Table 'data' est creee avec succes")
        except sqlite3.Error as e:
            print(f"Erreur de creation de table: {e}")

    def insertion(self, numbers):
        try:
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d %H:%M:%S")

            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO data (numbers, date_time)
                VALUES (?, ?)
            ''', (numbers, date_time))
            self.conn.commit()
            print("Insertion avec succes.")
        except sqlite3.Error as e:
            print(f"Erreur d'insertion des donnees: {e}")

    def recup(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''SELECT * FROM data''')
            results = cursor.fetchall()
            return results
        except sqlite3.Error as e:
            print(f"Erreur dans la recuperation des donnees: {e}")
            return None
       
    def update_plate(self, old_plate, new_plate):
        try:
            cursor = self.conn.cursor()
            cursor.execute("UPDATE data SET numbers = ? WHERE numbers = ?", (new_plate, old_plate))
            self.conn.commit()
            print("Mise a jour reussie.")
        except sqlite3.Error as e:
            print(f"Erreur lors de la mise a jour des donnees: {e}")

    def delete_plate(self, plate_text):
        try:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM data WHERE numbers = ?", (plate_text,))
            self.conn.commit()
            print("Suppression reussie.")
        except sqlite3.Error as e:
            print(f"Erreur lors de la suppression des donnees: {e}")

    def close_connection(self):
        if self.conn:
            self.conn.close()
            print(f"Connexion a la base de donnees '{self.db_name}' fermee.")

	
