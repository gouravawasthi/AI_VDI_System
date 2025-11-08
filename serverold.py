# server.py
from flask import Flask, jsonify,request
import sqlite3
import os
import datetime
import json

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
               d.front, d.back, d.right_side, d.left_side, d.top, d.down,d.station
        FROM checked_table AS c
        INNER JOIN inspection_details AS d
        ON c.barcode_id = d.barcode_id
        WHERE c.barcode_id = ?
        ORDER BY d.SessionId DESC LIMIT 1;
    """, (barcode_id,), one=True)

    if duplicate_check is not None:
        # Extract existing data
        status = duplicate_check[1]  # column 1 = c.status

        # Prepare detailed inspection info
        inspection_data = {
            "front": duplicate_check[2],
            "back": duplicate_check[3],
            "right_side": duplicate_check[4],
            "left_side": duplicate_check[5],
            "top": duplicate_check[6],
            "down": duplicate_check[7],
            "station":duplicate_check[8]  # New station column,
        }

        return jsonify({
            "barcode_id": barcode_id,
            "exists": True,
            "status": status,
            "message": "Barcode already inspected. Duplicate entry not allowed.",
            "inspection_details": inspection_data
        }), 401

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
LOG_DIR = "logs"  # Folder to store logs

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

def execute_db(query, args=()):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query, args)
    conn.commit()
    conn.close()
@app.route("/receive_log", methods=["POST"])
def receive_log():
    """Receive inspection JSON log and update database accordingly."""
    data = request.get_json(force=True)
    barcode_id = data.get("barcode_id")
    log_type = data.get("type", 0)
    steps = data.get("steps", [])
    final_status = data.get("final_status", "PENDING")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not barcode_id:
        return jsonify({"error": "Missing barcode_id"}), 400

    try:
        # --- Prepare inspection summary fields ---
        side_results = {s["step"].lower(): s["result"] for s in steps}
        front = side_results.get("front", None)
        back = side_results.get("back", None)
        left_side = side_results.get("left", None)
        right_side = side_results.get("right", None)
        top = side_results.get("top", None)
        down = side_results.get("bottom", None)
        station =data.get("station",None)

        # 1️⃣ TYPE = 1 (replace recent record for same barcode)
        if log_type == 1:
            print(f"🔁 Replacing last record for barcode {barcode_id}")
            execute_db(
                """
                DELETE FROM inspection_details
                WHERE rowid = (
                    SELECT rowid FROM inspection_details
                    WHERE barcode_id = ?
                    ORDER BY rowid DESC
                    LIMIT 1
                )
                """,
                (barcode_id,)
            )
            execute_db(
                """
                INSERT INTO inspection_details 
                (SessionID,barcode_id, front, back, left_side, right_side, top, down,station)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?,?)
                """,
                (timestamp,barcode_id, front, back, left_side, right_side, top, down,station)
            )
            return jsonify({
                "message": f"Inspection record replaced for {barcode_id}",
                "type": log_type,
                "status": final_status
            }), 200
        elif log_type in [0, 2]:
        # 2️⃣ TYPE = 0 or 2 (append new entry)
            print(f"📝 Inserting new record for barcode {barcode_id} [type={log_type}]")
            execute_db(
                """
                INSERT INTO inspection_details 
                (SessionID,barcode_id, front, back, left_side, right_side, top, down,station)
                VALUES (?, ?, ?, ?, ?, ?, ?,?,?)
                """,
                (timestamp,barcode_id, front, back, left_side, right_side, top, down,station)
            )

            return jsonify({
                "message": f"Inspection record updated for {barcode_id}",
                "type": log_type,
                "status": final_status
            }), 200

    except Exception as e:
        print(f"❌ Database update error: {e}")
        return jsonify({"error": str(e)}), 500


# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)