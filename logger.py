import json
import os
import requests
from datetime import datetime

class JSONLogger:
    def __init__(self, barcode_id,station,log_file="inspection_log.json"):
        self.log_file = log_file
        self.log_data = {
            "station": station,
            "barcode_id": barcode_id,
            "timestamp": str(datetime.now()),
            "steps": [],
            "final_status": None,
            "type": 0
                # 0=normal, 1=append, 2=update
        }

    def log_step(self, step_name, result):
        """
        Appends a step's result to the JSON log.
        step_name : str → e.g. 'Barcode Validation'
        result : str → e.g. 'PASS', 'FAIL', 'SKIPPED'
        details : dict/str → optional additional info
        """
        entry = {
            "step": step_name,
            "result": result,
            "timestamp": str(datetime.now())
        }
        self.log_data["steps"].append(entry)
        self._write_log()
    def set_type(self, action_type):
        """Set the type of operation: 0=normal, 1=append, 2=update"""
        if action_type in [0, 1, 2]:
            self.log_data["type"] = action_type
            self._write_log()
        else:
            print(f"⚠️ Invalid log type: {action_type}")

    def set_final_status(self, status):
        """Set final status after all steps (e.g., PASS or FAIL)."""
        self.log_data["final_status"] = status
        self.log_data["completed_at"] = str(datetime.now())
        self._write_log()

    def _write_log(self):
        """Save current log_data to file incrementally."""
        with open(self.log_file, "w") as f:
            json.dump(self.log_data, f, indent=4)

    def send_log_if_failed(self, server_url):
        """Send log to server only if any step failed."""
        for step in self.log_data["steps"]:
            if step["result"] == "FAIL":
                self._send_to_server(server_url)
                return True
        return False  # no failure

    def send_log_on_complete(self, server_url):
        """Send log to server when process fully completes."""
        self._send_to_server(server_url)

    def _send_to_server(self, server_url):
        """Internal: POST JSON log to server."""
        
        try:
            response = requests.post(server_url, json=self.log_data)
            if response.status_code == 200:
                print("✅ Log successfully sent to server.")
            else:
                print(f"⚠️ Failed to send log: {response.status_code}")
        except Exception as e:
            print(f"❌ Error sending log: {e}")

