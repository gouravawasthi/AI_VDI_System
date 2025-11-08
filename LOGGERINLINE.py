import json
import os
import requests
from datetime import datetime


class JSONLogger:
    """
    Unified inspection logger compatible with the new dynamic inspection API.
    Logs every inspection step locally and optionally pushes to server.
    """

    def __init__(self, barcode_id, station, process_name, log_dir="logs"):
        """
        Parameters:
            barcode_id : str
            station : str or int
            process_name : str - e.g., 'CHIP_INSPECTION', 'INLINE_INSPECTION_TOP'
            log_dir : str - directory to store per-barcode logs
        """
        self.barcode_id = barcode_id
        self.station = station
        self.process_name = process_name.upper()
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{barcode_id}_{process_name}_{timestamp}.json")

        self.log_data = {
            "barcode_id": barcode_id,
            "station": station,
            "process": process_name,
            "timestamp": str(datetime.now()),
            "steps": [],
            "payloads": [],
            "final_status": None,
            "type": 0,   # 0=normal, 1=append, 2=update
        }
        self._write_log()

    # -------------------------------------------------------
    # Step Logging
    # -------------------------------------------------------
    def log_step(self, step_name, result, details=None):
        """
        Logs a single step during the inspection.

        Parameters:
            step_name : str - name of the stage, e.g. "TOP Capture"
            result : str - 'PASS', 'FAIL', or 'SKIPPED'
            details : dict or str - optional metadata (time, error, etc.)
        """
        entry = {
            "step": step_name,
            "result": result,
            "timestamp": str(datetime.now())
        }
        if details is not None:
            entry["details"] = details
        self.log_data["steps"].append(entry)
        self._write_log()

    # -------------------------------------------------------
    # Payload Logging
    # -------------------------------------------------------
    def log_payload(self, payload: dict):
        """Store API payload (e.g., the inspection result POSTed to /api)."""
        self.log_data["payloads"].append({
            "timestamp": str(datetime.now()),
            "payload": payload
        })
        self._write_log()

    # -------------------------------------------------------
    # Operation Type
    # -------------------------------------------------------
    def set_type(self, action_type: int):
        """
        Set operation type (0=normal, 1=append, 2=update)
        """
        if action_type in [0, 1, 2]:
            self.log_data["type"] = action_type
            self._write_log()
        else:
            print(f"⚠️ Invalid log type: {action_type}")

    # -------------------------------------------------------
    # Final Result
    # -------------------------------------------------------
    def set_final_status(self, status: str):
        """Set final inspection result ('PASS' or 'FAIL')."""
        self.log_data["final_status"] = status
        self.log_data["completed_at"] = str(datetime.now())
        self._write_log()

    # -------------------------------------------------------
    # Local Write
    # -------------------------------------------------------
    def _write_log(self):
        """Write current log state to JSON file."""
        with open(self.log_path, "w") as f:
            json.dump(self.log_data, f, indent=4)

    # -------------------------------------------------------
    # Server Sync
    # -------------------------------------------------------
    def send_to_server(self, base_api_url="http://127.0.0.1:5000/api"):
        """
        Sends the last logged payload to the correct inspection endpoint.
        For example:
            - CHIP_INSPECTION → /api/CHIPINSPECTION
            - INLINE_INSPECTION_TOP → /api/INLINEINSPECTIONTOP
        """
        if not self.log_data["payloads"]:
            print("⚠️ No payload to send.")
            return False

        last_payload = self.log_data["payloads"][-1]["payload"]
        api_url = f"{base_api_url}/{self.process_name}"

        try:
            response = requests.post(api_url, json=last_payload, timeout=5)
            if response.status_code in (200, 201):
                print(f"✅ Log and payload sent successfully to {self.process_name}")
                return True
            else:
                print(f"⚠️ Server responded with {response.status_code}: {response.text}")
                return False
        except requests.ConnectionError:
            print(f"❌ Connection Error: Could not reach {api_url}")
            return False
        except Exception as e:
            print(f"❌ Error sending to server: {e}")
            return False
