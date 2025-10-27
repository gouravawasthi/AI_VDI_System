"""
Barcode Handler - Manages barcode scanner input
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, List
from queue import Queue, Empty

try:
    import pyzbar.pyzbar as pyzbar
    from pyzbar import ZBarSymbol
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

import cv2
import numpy as np


class BarcodeHandler:
    """
    Handles barcode reading operations for the AI VDI System
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize barcode handler
        
        Args:
            config: Barcode configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.scan_timeout = config.get('scan_timeout', 5.0)
        self.supported_formats = config.get('supported_formats', ['CODE128', 'QR'])
        
        self.barcode_queue = Queue()
        self.is_scanning = False
        self.scan_thread = None
        self.logger = self._setup_logger()
        
        if not PYZBAR_AVAILABLE and self.enabled:
            self.logger.warning("pyzbar not available. Barcode functionality disabled.")
            self.enabled = False
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for barcode handler"""
        logger = logging.getLogger('BarcodeHandler')
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
        Initialize barcode scanner
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.enabled:
            self.logger.info("Barcode scanning disabled")
            return True
        
        if not PYZBAR_AVAILABLE:
            self.logger.error("pyzbar library not available")
            return False
        
        try:
            self.logger.info("Barcode handler initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize barcode handler: {e}")
            return False
    
    def scan_from_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Scan for barcodes in an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of detected barcodes with their data
        """
        if not self.enabled or not PYZBAR_AVAILABLE:
            return []
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Decode barcodes
            barcodes = pyzbar.decode(gray)
            
            results = []
            for barcode in barcodes:
                # Extract barcode data
                barcode_data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                
                # Get bounding box
                x, y, w, h = barcode.rect
                
                result = {
                    'data': barcode_data,
                    'type': barcode_type,
                    'bbox': (x, y, w, h),
                    'timestamp': time.time()
                }
                
                results.append(result)
                self.logger.info(f"Detected {barcode_type}: {barcode_data}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error scanning barcode from image: {e}")
            return []
    
    def read_barcode(self, timeout: Optional[float] = None) -> Optional[str]:
        """
        Read barcode data (blocking call)
        
        Args:
            timeout: Timeout in seconds (None for default)
            
        Returns:
            Barcode data string or None if no barcode found
        """
        if not self.enabled:
            return None
        
        try:
            timeout = timeout or self.scan_timeout
            barcode_data = self.barcode_queue.get(timeout=timeout)
            return barcode_data
            
        except Empty:
            self.logger.debug("No barcode data available within timeout")
            return None
        except Exception as e:
            self.logger.error(f"Error reading barcode: {e}")
            return None
    
    def start_continuous_scan(self, camera_handler=None):
        """
        Start continuous barcode scanning in background thread
        
        Args:
            camera_handler: Camera handler instance for image capture
        """
        if not self.enabled or self.is_scanning:
            return
        
        self.is_scanning = True
        self.scan_thread = threading.Thread(
            target=self._continuous_scan_loop,
            args=(camera_handler,)
        )
        self.scan_thread.start()
        self.logger.info("Started continuous barcode scanning")
    
    def stop_continuous_scan(self):
        """Stop continuous barcode scanning"""
        if not self.is_scanning:
            return
        
        self.is_scanning = False
        if self.scan_thread:
            self.scan_thread.join(timeout=5.0)
        
        self.logger.info("Stopped continuous barcode scanning")
    
    def _continuous_scan_loop(self, camera_handler):
        """
        Continuous scanning loop (runs in separate thread)
        
        Args:
            camera_handler: Camera handler instance
        """
        while self.is_scanning:
            try:
                if camera_handler:
                    # Capture image from camera
                    image = camera_handler.capture_image()
                    if image is not None:
                        # Scan for barcodes
                        barcodes = self.scan_from_image(image)
                        
                        # Add detected barcodes to queue
                        for barcode in barcodes:
                            try:
                                self.barcode_queue.put_nowait(barcode['data'])
                            except:
                                pass  # Queue full, skip
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in continuous scan loop: {e}")
                time.sleep(1.0)  # Longer delay on error
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported barcode formats
        
        Returns:
            List of supported format names
        """
        if not PYZBAR_AVAILABLE:
            return []
        
        # Map ZBar symbols to format names
        format_map = {
            ZBarSymbol.CODE128: 'CODE128',
            ZBarSymbol.CODE39: 'CODE39',
            ZBarSymbol.CODE93: 'CODE93',
            ZBarSymbol.CODABAR: 'CODABAR',
            ZBarSymbol.DATABAR: 'DATABAR',
            ZBarSymbol.DATABAR_EXP: 'DATABAR_EXP',
            ZBarSymbol.EAN13: 'EAN13',
            ZBarSymbol.EAN8: 'EAN8',
            ZBarSymbol.I25: 'I25',
            ZBarSymbol.ISBN10: 'ISBN10',
            ZBarSymbol.ISBN13: 'ISBN13',
            ZBarSymbol.PDF417: 'PDF417',
            ZBarSymbol.QRCODE: 'QR',
            ZBarSymbol.UPCA: 'UPCA',
            ZBarSymbol.UPCE: 'UPCE'
        }
        
        return list(format_map.values())
    
    def clear_queue(self):
        """Clear the barcode queue"""
        while not self.barcode_queue.empty():
            try:
                self.barcode_queue.get_nowait()
            except Empty:
                break
    
    def cleanup(self):
        """Cleanup barcode handler resources"""
        try:
            self.stop_continuous_scan()
            self.clear_queue()
            self.logger.info("Barcode handler cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()