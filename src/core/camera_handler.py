"""
Camera Handler - Manages camera input using OpenCV
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple


class CameraHandler:
    """
    Handles camera operations for the AI VDI System
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize camera handler
        
        Args:
            config: Camera configuration dictionary
        """
        self.config = config
        self.camera_id = config.get('camera_id', 0)
        self.resolution = config.get('resolution', (1920, 1080))
        self.fps = config.get('fps', 30)
        
        self.cap = None
        self.is_initialized = False
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for camera handler"""
        logger = logging.getLogger('CameraHandler')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize(self) -> bool:
        """
        Initialize camera connection
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def capture_image(self) -> Optional[np.ndarray]:
        """
        Capture a single image from the camera
        
        Returns:
            numpy.ndarray: Captured image or None if capture failed
        """
        if not self.is_initialized or self.cap is None:
            self.logger.error("Camera not initialized")
            return None
        
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                self.logger.error("Failed to capture frame")
                return None
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error capturing image: {e}")
            return None
    
    def capture_continuous(self):
        """
        Generator function for continuous frame capture
        
        Yields:
            numpy.ndarray: Captured frames
        """
        if not self.is_initialized or self.cap is None:
            self.logger.error("Camera not initialized")
            return
        
        while True:
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.error("Failed to capture frame")
                    break
                
                yield frame
                
            except Exception as e:
                self.logger.error(f"Error in continuous capture: {e}")
                break
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera information and current settings
        
        Returns:
            Dict containing camera information
        """
        if not self.is_initialized or self.cap is None:
            return {}
        
        try:
            info = {
                'camera_id': self.camera_id,
                'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'backend': self.cap.getBackendName(),
                'is_opened': self.cap.isOpened()
            }
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting camera info: {e}")
            return {}
    
    def set_property(self, property_id: int, value: float) -> bool:
        """
        Set camera property
        
        Args:
            property_id: OpenCV property ID
            value: Property value to set
            
        Returns:
            bool: True if property set successfully, False otherwise
        """
        if not self.is_initialized or self.cap is None:
            self.logger.error("Camera not initialized")
            return False
        
        try:
            result = self.cap.set(property_id, value)
            if result:
                self.logger.info(f"Set property {property_id} to {value}")
            else:
                self.logger.warning(f"Failed to set property {property_id} to {value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error setting property: {e}")
            return False
    
    def get_property(self, property_id: int) -> Optional[float]:
        """
        Get camera property value
        
        Args:
            property_id: OpenCV property ID
            
        Returns:
            Property value or None if failed
        """
        if not self.is_initialized or self.cap is None:
            self.logger.error("Camera not initialized")
            return None
        
        try:
            return self.cap.get(property_id)
            
        except Exception as e:
            self.logger.error(f"Error getting property: {e}")
            return None
    
    def cleanup(self):
        """Release camera resources"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
            self.is_initialized = False
            self.logger.info("Camera resources released")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()