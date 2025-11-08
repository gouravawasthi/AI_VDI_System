import requests
from typing import Any, Dict, Optional, Tuple

# --- API Endpoint Constants ---
# Base URL for the inspection API server
API_BASE_URL = "http://127.0.0.1:5001/api"

# Database table endpoints - based on inspection_data.db tables
API_ENDPOINTS = {
    'CHIP_INSPECTION': f"{API_BASE_URL}/CHIPINSPECTION",
    'INLINE_INSPECTION_TOP': f"{API_BASE_URL}/INLINEINSPECTIONTOP", 
    'INLINE_INSPECTION_BOTTOM': f"{API_BASE_URL}/INLINEINSPECTIONBOTTOM",
    'EOLT_INSPECTION': f"{API_BASE_URL}/EOLTINSPECTION"
}

# Pre-configured API pairs for common inspection workflows
INSPECTION_WORKFLOWS = {
    'CHIP_TO_EOLT': {
        'api1_url': API_ENDPOINTS['CHIP_INSPECTION'],
        'api2_url': API_ENDPOINTS['EOLT_INSPECTION'],
        'placeholders': ('chip inspection', 'EOLT testing')
    },
    'INLINE_TOP_TO_EOLT': {
        'api1_url': API_ENDPOINTS['INLINE_INSPECTION_TOP'],
        'api2_url': API_ENDPOINTS['EOLT_INSPECTION'], 
        'placeholders': ('inline top inspection', 'EOLT testing')
    },
    'INLINE_BOTTOM_TO_EOLT': {
        'api1_url': API_ENDPOINTS['INLINE_INSPECTION_BOTTOM'],
        'api2_url': API_ENDPOINTS['EOLT_INSPECTION'],
        'placeholders': ('inline bottom inspection', 'EOLT testing')
    },
    'CHIP_TO_INLINE_TOP': {
        'api1_url': API_ENDPOINTS['CHIP_INSPECTION'],
        'api2_url': API_ENDPOINTS['INLINE_INSPECTION_TOP'],
        'placeholders': ('chip inspection', 'inline top inspection')
    },
    'CHIP_TO_INLINE_BOTTOM': {
        'api1_url': API_ENDPOINTS['CHIP_INSPECTION'],
        'api2_url': API_ENDPOINTS['INLINE_INSPECTION_BOTTOM'],
        'placeholders': ('chip inspection', 'inline bottom inspection')
    },
    'INLINE_TOP_TO_INLINE_BOTTOM': {
        'api1_url': API_ENDPOINTS['INLINE_INSPECTION_TOP'],
        'api2_url': API_ENDPOINTS['INLINE_INSPECTION_BOTTOM'],
        'placeholders': ('inline top inspection', 'inline bottom inspection')
    },
    'INLINE_BOTTOM_TO_INLINE_TOP': {
        'api1_url': API_ENDPOINTS['INLINE_INSPECTION_BOTTOM'],
        'api2_url': API_ENDPOINTS['INLINE_INSPECTION_TOP'],
        'placeholders': ('inline bottom inspection', 'inline top inspection')
    }   
}


class APIManager:
    """
    Handles two dependent API calls (API1 and API2) and subsequent actions:
    delete, update, append.

    Usage:
        api = APIManager(api1_url, api2_url, placeholders=("visual", "electrical"))
        result = api.process_barcode("12345")
        if result["action_required"]:
            api.execute_action("delete", barcode="12345")
    """

    def __init__(self, api1_url: str, api2_url: str, placeholders: Tuple[str, str]):
        self.api1_url = api1_url
        self.api2_url = api2_url
        self.placeholders = placeholders  # for user messages
        self.pending_actions: Dict[str, Dict[str, Any]] = {}  # store future update/append payloads

    @classmethod
    def create_workflow(cls, workflow_name: str) -> 'APIManager':
        """
        Factory method to create APIManager from predefined workflows.
        
        Args:
            workflow_name: Name of predefined workflow (e.g., 'CHIP_TO_EOLT')
            
        Returns:
            APIManager instance configured for the workflow
            
        Example:
            api_manager = APIManager.create_workflow('CHIP_TO_EOLT')
        """
        if workflow_name not in INSPECTION_WORKFLOWS:
            available = list(INSPECTION_WORKFLOWS.keys())
            raise ValueError(f"Unknown workflow '{workflow_name}'. Available workflows: {available}")
        
        workflow_config = INSPECTION_WORKFLOWS[workflow_name]
        return cls(
            api1_url=workflow_config['api1_url'],
            api2_url=workflow_config['api2_url'],
            placeholders=workflow_config['placeholders']
        )

    @classmethod  
    def create_from_config(cls, api1_endpoint: str, api2_endpoint: str, placeholders: Tuple[str, str]) -> 'APIManager':
        """
        Factory method to create APIManager from endpoint names.
        
        Args:
            api1_endpoint: First API endpoint name (e.g., 'CHIP_INSPECTION')
            api2_endpoint: Second API endpoint name (e.g., 'EOLT_INSPECTION')
            placeholders: Description tuple for user messages
            
        Returns:
            APIManager instance
            
        Example:
            api_manager = APIManager.create_from_config(
                'CHIP_INSPECTION', 
                'EOLT_INSPECTION',
                ('chip inspection', 'EOLT testing')
            )
        """
        if api1_endpoint not in API_ENDPOINTS:
            available = list(API_ENDPOINTS.keys())
            raise ValueError(f"Unknown endpoint '{api1_endpoint}'. Available endpoints: {available}")
            
        if api2_endpoint not in API_ENDPOINTS:
            available = list(API_ENDPOINTS.keys())  
            raise ValueError(f"Unknown endpoint '{api2_endpoint}'. Available endpoints: {available}")
        
        return cls(
            api1_url=API_ENDPOINTS[api1_endpoint],
            api2_url=API_ENDPOINTS[api2_endpoint],
            placeholders=placeholders
        )

    # ---------------------------------------------------
    #  Internal API caller helper
    # ---------------------------------------------------
    def _call_api(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: int = 5,
    ) -> Tuple[bool, Optional[Any]]:
        try:
            method = method.lower()
            if method == "get":
                response = requests.get(url, timeout=timeout)
            elif method == "post":
                response = requests.post(url, json=payload, timeout=timeout)
            elif method == "put":
                response = requests.put(url, json=payload, timeout=timeout)
            elif method == "delete":
                response = requests.delete(url, json=payload, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            if response.status_code not in (200, 201, 204, 404):
                return False, None

            try:
                return True, response.json()
            except ValueError:
                return True, None

        except requests.exceptions.ConnectionError:
            return False, None
        except Exception as e:
            print(f"[APIManager] Error calling {url} ({method}): {e}")
            return False, None

    # ---------------------------------------------------
    #  Main process logic
    # ---------------------------------------------------
    def process_barcode(self, barcode: str) -> Dict[str, Any]:
        """
        Main logic orchestrating both APIs.
        Returns structured dict for GUI.
        """

        msg = {
            "status": "error",
            "message": "",
            "buttons": [],
            "data": None,
            "action_required": False,
        }

        ok1, result1 = self._call_api("get", f"{self.api1_url}?barcode={barcode}")
        ok2, result2 = self._call_api("get", f"{self.api2_url}?barcode={barcode}")

        # ---- Server availability ----
        if not ok1 and not ok2:
            msg["message"] = "Can't proceed — both servers are not running."
            return msg
        elif not ok1:
            msg["message"] = f"Can't proceed — Server {self.placeholders[0]} not running."
            return msg
        elif not ok2:
            msg["message"] = f"Can't proceed — Server {self.placeholders[1]} not running."
            return msg

        # Evaluate API responses to get boolean results
        result1_eval = self._evaluate_manual_result(result1)
        result2_eval = self._evaluate_manual_result(result2)

        # ---- API1 None ----
        if result1 in (None, "", {}, []):
            msg["message"] = (
                f"Barcode was not tested in previous {self.placeholders[0]} inspection, can’t proceed."
            )
            return msg

        # ---- API1 True ----
        if result1 is True:
            # API2 None → proceed with new entry (will do POST request after collecting results)
            if result2 in (None, "", {}, []):
                return {
                    "status": "success",
                    "message": f"Proceed with {self.placeholders[1]} - New entry will be created.",
                    "buttons": [],
                    "data": None,
                    "action_required": False,
                }

            # API2 contains duplicate → ask user
            if isinstance(result2, dict) and len(result2) > 0:
                msg.update(
                    {
                        "status": "warning",
                        "message": f"Barcode already scanned in {self.placeholders[1]} — duplicate record. Do you want to proceed?",
                        "buttons": ["Delete", "Update", "Append"],
                        "data": result2,
                        "action_required": True,
                    }
                )
                return msg

        msg["message"] = "Unexpected API response."
        return msg
    
    def _evaluate_manual_result(self, api_response: Any) -> Optional[bool]:
        """
        Evaluate API response and extract ManualResult or PASS_FAIL value.
        
        Args:
            api_response: Response from API call
            
        Returns:
            True if passed (ManualResult=1 or PASS_FAIL=1)
            False if failed (ManualResult=0 or PASS_FAIL=0)  
            None if no record found or invalid response
        """
        if not api_response:
            return None
            
        # Handle error responses (404, etc.)
        if isinstance(api_response, dict) and "message" in api_response:
            if "No record found" in api_response["message"]:
                return None
            return None
            
        # Handle successful responses with data
        if isinstance(api_response, dict) and "data" in api_response:
            data_list = api_response["data"]
            if not data_list or len(data_list) == 0:
                return None
                
            # Get the record (latest if multiple)
            record = data_list[0] if isinstance(data_list, list) else data_list
            
            if isinstance(record, dict):
                # Check ManualResult first (priority)
                if "ManualResult" in record:
                    return record["ManualResult"] == 1
                    
                # Fallback to PASS_FAIL
                if "PASS_FAIL" in record:
                    return record["PASS_FAIL"] == 1
                    
        return None

    # ---------------------------------------------------
    #  Action handlers (delete, update, append)
    # ---------------------------------------------------
    def execute_action(self, action: str, barcode: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute user-chosen action:
          - delete: immediate DELETE call to API2
          - update/append: store payload for later POST/PUT
        """
        result = {"status": "error", "message": "", "data": None}

        action = action.lower().strip()

        if action == "delete":
            ok, resp = self._call_api("delete", self.api2_url, {"barcode": barcode})
            if ok:
                result["status"] = "success"
                result["message"] = f"Barcode {barcode} deleted successfully."
                result["data"] = resp
            else:
                result["message"] = f"Failed to delete barcode {barcode}."
            return result

        elif action in ("update", "append"):
            # save for later execution
            self.pending_actions[barcode] = {"action": action, "data": data}
            result["status"] = "pending"
            result["message"] = f"Action '{action}' recorded for barcode {barcode}. Will execute after inspection."
            return result

        else:
            result["message"] = f"Invalid action '{action}'."
            return result

    # ---------------------------------------------------
    #  Trigger pending actions after inspection
    # ---------------------------------------------------
    def commit_pending_actions(self) -> Dict[str, Any]:
        """
        Execute all stored update/append actions.
        Returns summary.
        """
        summary = {"executed": 0, "failed": 0}

        for barcode, action_info in list(self.pending_actions.items()):
            action = action_info["action"]
            data = action_info["data"]

            if action == "update":
                ok, _ = self._call_api("put", self.api2_url, {"barcode": barcode, "data": data})
            elif action == "append":
                ok, _ = self._call_api("post", self.api2_url, {"barcode": barcode, "data": data})
            else:
                ok = False

            if ok:
                summary["executed"] += 1
                del self.pending_actions[barcode]
            else:
                summary["failed"] += 1

        return summary

#from api_manager import APIManager

#api = APIManager(
#    api1_url="http://localhost:5001/check_previous",
#    api2_url="http://localhost:5002/check_duplicate",
#    placeholders=("visual", "electrical")
#)

## --- Validate barcode ---
#response = api.process_barcode("ABC123")

#if response["status"] == "warning":
#    # GUI asks user for Delete / Update / Append
#    user_action = "delete"
#    action_result = api.execute_action(user_action, "ABC123")
#    print(action_result)

#elif response["status"] == "success":
#    print("Proceed to inspection...")

## --- Later (after inspection data collected) ---
#final = api.commit_pending_actions()
#print("Pending updates summary:", final)
