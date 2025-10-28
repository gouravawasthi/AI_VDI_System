# init_db.py
import sqlite3

conn = sqlite3.connect("inspection.db")
cursor = conn.cursor()

cursor.execute("""ALTER TABLE inspection_details ADD COLUMN station INTEGER DEFAULT 1;

""")
conn.commit()

# Close the connection
conn.close()