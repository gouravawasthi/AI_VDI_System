# FILE: automation_script.py

import sqlite3
import pandas as pd
import os

# --- Configuration (Modified for data/db and src/server directory structure) ---
# BASE_DIR is the directory from which the script is executed
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Directory for the DB and CSV files (data/db)
DATA_DIR = os.path.join(BASE_DIR, "data", "db") 

# Path to the database file (inside data/db)
DB_FILE = os.path.join(DATA_DIR, "inspection_data.db")

# Path to the SQL schema file (inside src/server)
SQL_SCRIPT_PATH = os.path.join(BASE_DIR, "src", "server", "schema.sql") 

# Dictionary mapping table names to their respective dummy CSV file names
# NOTE: CSV files MUST be placed inside the data/db directory!
CSV_MAPPING = {
    'CHIPINSPECTION': 'chip_data_dummy.csv',
    'INLINEINSPECTIONBOTTOM': 'inline_bottom_dummy.csv',
    'INLINEINSPECTIONTOP': 'inline_top_dummy.csv',
    'EOLTINSPECTION': 'eolt_dummy.csv',
}

# --- Functions ---

def create_database_and_tables(db_file, sql_script):
    """Creates the SQLite DB and tables from the schema script."""
    # 1. Ensure the DB directory exists (data/db)
    if not os.path.exists(DATA_DIR):
        print(f"Creating data directory: {DATA_DIR}")
        os.makedirs(DATA_DIR)
    
    # 2. Check if the SQL schema file exists in its new location
    if not os.path.exists(sql_script):
        print(f"ERROR: SQL schema file not found at expected path: {sql_script}")
        return None
        
    print(f"--- Connecting/Creating Database at {db_file} ---")
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        with open(sql_script, 'r') as f:
            sql_commands = f.read()
        
        cursor.executescript(sql_commands)
        conn.commit()
        print("Schema loaded successfully: Tables and Indexes created.")
        return conn
    except sqlite3.Error as e:
        print(f"ERROR: SQLite database operation failed: {e}")
        return None

def insert_data_from_csv(conn, table_name, csv_filename):
    """Loads data from a CSV file (expected in data/db) into the specified SQLite table."""
    csv_file_path = os.path.join(DATA_DIR, csv_filename)
    
    if not os.path.exists(csv_file_path):
        print(f"Skipping {table_name}: CSV file '{csv_file_path}' not found. Please create it and place it in the {os.path.relpath(DATA_DIR, start=BASE_DIR)} folder.")
        return

    print(f"\n--- Loading data into {table_name} from {csv_filename} ---")
    try:
        df = pd.read_csv(csv_file_path)
        
        # Robustness: Convert relevant columns to integer for SQLite's BOOLEAN (1/0)
        #int_cols = [col for col in df.columns if any(keyword in col.upper() for keyword in ['ID', 'INT', 'FAIL', 'MANUAL', 'RESULT'])]
        #for col in int_cols:
        #     if col in df.columns:
        #        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        df.to_sql(table_name, conn, if_exists='append', index=False)
        conn.commit()
        print(f"SUCCESS: {len(df)} records loaded into {table_name}.")

    except Exception as e:
        print(f"ERROR: Failed to load data for {table_name}: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    # 1. Run checks
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas library not found. Please install it using: pip install pandas")
        exit(1)

    # 2. Create Database and Tables
    db_connection = create_database_and_tables(DB_FILE, SQL_SCRIPT_PATH)
    
    if db_connection:
        # 3. Insert Dummy Records for Testing
        print("\n==============================================")
        print("Starting Dummy Data Insertion...")
        print("==============================================")

        for table, csv_file in CSV_MAPPING.items():
            insert_data_from_csv(db_connection, table, csv_file)
        
        # 4. Cleanup
        db_connection.close()
        print("\nAutomation finished. Database connection closed.")