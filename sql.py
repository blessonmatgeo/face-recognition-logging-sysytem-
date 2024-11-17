import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('data/faces_database.db')
cursor = conn.cursor()

# Create a table to store face features if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_number TEXT,
        name TEXT,
        feature_1 REAL,
        feature_2 REAL,
        ...
        feature_128 REAL
    )
''')

class Face_Register:
    # Existing code...

    def extract_and_save_features(self, path_images_from_camera):
        # Existing code...

        for person in person_list:
            features_mean_personX = self.return_features_mean_personX(path_images_from_camera + person)
            if len(person.split('_', 2)) == 2:
                employee_number, person_name = person.split('_', 2)
            else:
                employee_number, person_name = None, person.split('_', 2)[-1]

            # Prepare SQL query to insert data into the table
            sql_query = '''INSERT INTO faces (employee_number, name, feature_1, feature_2, ..., feature_128) 
                           VALUES (?, ?, ?, ?, ..., ?)'''
            data = [employee_number, person_name] + list(features_mean_personX)
            
            # Execute SQL query to insert data
            cursor.execute(sql_query, data)
            conn.commit()

        logging.info("Saved all the features of faces registered into: data/faces_database.db")

    # Existing code...

    def run(self):
        # Existing code...
        conn.close()  # Close the connection when the program terminates
