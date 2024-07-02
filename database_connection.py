import sqlite3

DB_PATH = 'car_license/malaysia_license_plates.db'

def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS LicensePlates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL UNIQUE
        )
    ''')
    plates = ['ABC1234', 'WXYZ5678', 'JKL9876', 'MNOP4321',
              'WTL9184','PPS5988']
    for plate in plates:
        c.execute('INSERT OR IGNORE INTO LicensePlates (plate) VALUES (?)', (plate,))
    conn.commit()
    conn.close()

def check_license_plate(plate):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT plate FROM LicensePlates WHERE plate = ?', (plate,))
    result = c.fetchone()
    conn.close()
    is_recorded = result is not None
    allow_entry = is_recorded  # 假设在数据库中即放行 Assume that it is allowed in the database
    return is_recorded, allow_entry

initialize_database()
