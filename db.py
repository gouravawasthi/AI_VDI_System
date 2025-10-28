# init_db.py
import sqlite3

conn = sqlite3.connect("inspection.db")
cursor = conn.cursor()

cursor.execute(""" DELETE FROM inspection_details
WHERE barcode_id IN ('A123', 'B456');
""")
conn.commit()

# Close the connection
conn.close()