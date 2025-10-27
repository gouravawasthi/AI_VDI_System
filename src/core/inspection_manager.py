"""
Inspection Manager - Orchestrates ML, camera, and barcode operations
"""

import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any

try:
    from .camera_handler import CameraHandler
    from .barcode_handler import BarcodeHandler
    from ..ml.inference import InferenceEngine
except ImportError:
    # Fallback for direct execution from root
    from src.core.camera_handler import CameraHandler
    from src.core.barcode_handler import BarcodeHandler
    from src.ml.inference import InferenceEngine


class InspectionManager:
    """
    Orchestrates the inspection process by coordinating camera,
    barcode scanner, and ML inference components
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the inspection manager
        
        Args:
            config: Configuration dictionary containing system settings
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # Initialize components
        self.camera_handler = CameraHandler(config.get('camera', {}))
        self.barcode_handler = BarcodeHandler(config.get('barcode', {}))
        self.inference_engine = InferenceEngine(config.get('ml', {}))
        
        # State management
        self.is_running = False
        self.inspection_thread = None
        self.inspection_results = []
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the inspection manager"""
        logger = logging.getLogger('InspectionManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def start_inspection(self) -> bool:
        """
        Start the inspection process
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.is_running:
            self.logger.warning("Inspection already running")
            return False
        
        try:
            # Initialize components
            if not self.camera_handler.initialize():
                self.logger.error("Failed to initialize camera")
                return False
            
            if not self.barcode_handler.initialize():
                self.logger.error("Failed to initialize barcode scanner")
                return False
            
            if not self.inference_engine.load_model():
                self.logger.error("Failed to load ML model")
                return False
            
            # Start inspection thread
            self.is_running = True
            self.inspection_thread = threading.Thread(target=self._inspection_loop)
            self.inspection_thread.start()
            
            self.logger.info("Inspection started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start inspection: {e}")
            return False
    
    def stop_inspection(self) -> bool:
        """
        Stop the inspection process
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_running:
            self.logger.warning("Inspection not running")
            return False
        
        try:
            self.is_running = False
            
            if self.inspection_thread:
                self.inspection_thread.join(timeout=5.0)
            
            # Cleanup components
            self.camera_handler.cleanup()
            self.barcode_handler.cleanup()
            
            self.logger.info("Inspection stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop inspection: {e}")
            return False
    
    def _inspection_loop(self):
        """Main inspection loop running in separate thread"""
        self.logger.info("Starting inspection loop")
        
        while self.is_running:
            try:
                # Capture image from camera
                image = self.camera_handler.capture_image()
                if image is None:
                    continue
                
                # Read barcode if available
                barcode_data = self.barcode_handler.read_barcode()
                
                # Run ML inference
                prediction = self.inference_engine.predict(image)
                
                # Process results
                result = self._process_inspection_result(
                    image, barcode_data, prediction
                )
                
                # Store result
                self.inspection_results.append(result)
                
                # Log result
                self.logger.info(f"Inspection result: {result['status']}")
                
            except Exception as e:
                self.logger.error(f"Error in inspection loop: {e}")
    
    def _process_inspection_result(self, image, barcode_data, prediction) -> Dict[str, Any]:
        """
        Process inspection results and return formatted result
        
        Args:
            image: Captured image
            barcode_data: Barcode data if available
            prediction: ML prediction result
            
        Returns:
            Dict containing inspection result
        """
        timestamp = datetime.now().isoformat()
        
        # Determine pass/fail status based on prediction
        threshold = self.config.get('ml', {}).get('threshold', 0.5)
        is_defective = prediction.get('confidence', 0) > threshold
        
        result = {
            'timestamp': timestamp,
            'barcode': barcode_data,
            'prediction': prediction,
            'status': 'FAIL' if is_defective else 'PASS',
            'confidence': prediction.get('confidence', 0)
        }
        
        return result
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest inspection result
        
        Returns:
            Latest inspection result or None if no results available
        """
        if self.inspection_results:
            return self.inspection_results[-1]
        return None
    
    def get_inspection_history(self, limit: int = 100) -> list:
        """
        Get inspection history
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of inspection results
        """
        return self.inspection_results[-limit:] if self.inspection_results else []