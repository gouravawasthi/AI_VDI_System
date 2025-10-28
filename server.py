# server.py
from flask import Flask, jsonify
import sqlite3

app = Flask(__name__)
DB_PATH = "inspection.db"

def query_db(query, args=(), one=False):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query, args)
    rows = cursor.fetchall()
    conn.close()
    return (rows[0] if rows else None) if one else rows

@app.route('/validate/<barcode_id>', methods=['GET'])
def validate(barcode_id):
    """Check if barcode exists, duplicate, and inspection status"""
    
    # 1️⃣ Check if barcode already has an inspection record (duplicate check)
    duplicate_check = query_db("""
        SELECT c.barcode_id, c.status,
               d.front, d.back, d.right_side, d.left_side, d.top, d.down
        FROM checked_table AS c
        INNER JOIN inspection_details AS d
        ON c.barcode_id = d.barcode_id
        WHERE c.barcode_id = ?
    """, (barcode_id,), one=True)

    if duplicate_check is not None:
        # Extract existing status
        status = duplicate_check[1]  # column 1 = c.status
        return jsonify({
            "barcode_id": barcode_id,
            "exists": True,
            "status": status,
            "message": "Barcode already inspected. Duplicate entry not allowed."
        }), 401 # Conflict

    # 2️⃣ Check if barcode exists in main table
    row = query_db("SELECT status FROM checked_table WHERE barcode_id = ?", (barcode_id,), one=True)
    if not row:
        return jsonify({
            "exists": False,
            "message": "Barcode not found in database."
        }), 404

    # 3️⃣ Check if previously failed
    failed_check = query_db("""
        SELECT 1 FROM checked_table 
        WHERE barcode_id = ? AND status = 'FAIL'
    """, (barcode_id,), one=True)

    status = row[0]

    if failed_check is not None:
        return jsonify({
            "barcode_id": barcode_id,
            "exists": True,
            "status": status,
            "valid": False,
            "message": "Barcode previously failed inspection."
        }), 409  # Conflict

    # 4️⃣ Otherwise, valid for inspection
    return jsonify({
        "barcode_id": barcode_id,
        "exists": True,
        "status": status,
        "valid": True,
        "message": "Barcode is valid for inspection."
    }), 200
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
