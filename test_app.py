#!/usr/bin/env python3
"""
AI VDI System - Simple Test Version
Visual Defect Inspection System using AI/ML for automated quality control
"""

import sys
import os
import configparser
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton
from PyQt5.QtCore import Qt
import cv2
import numpy as np


class SimpleAIVDISystem(QMainWindow):
    """Simple version of the AI VDI System for testing"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup basic logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SimpleAIVDI')
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("AI VDI System - Test Version")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Add title
        title = QLabel("AI Visual Defect Inspection System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        layout.addWidget(title)
        
        # Add status
        self.status_label = QLabel("System Status: Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 16px; margin: 10px;")
        layout.addWidget(self.status_label)
        
        # Add camera test button
        self.camera_btn = QPushButton("Test Camera")
        self.camera_btn.clicked.connect(self.test_camera)
        self.camera_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.camera_btn)
        
        # Add package test button
        self.package_btn = QPushButton("Test Packages")
        self.package_btn.clicked.connect(self.test_packages)
        self.package_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(self.package_btn)
        
        # Add result display
        self.result_label = QLabel("Test results will appear here...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 14px; margin: 20px; padding: 20px; border: 1px solid gray;")
        layout.addWidget(self.result_label)
    
    def test_camera(self):
        """Test camera functionality"""
        self.logger.info("Testing camera...")
        self.status_label.setText("System Status: Testing Camera...")
        
        try:
            # Try to open camera
            cap = cv2.VideoCapture(0)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    result = f"✅ Camera test successful!\nResolution: {width}x{height}"
                    self.logger.info("Camera test passed")
                else:
                    result = "❌ Camera opened but failed to capture frame"
                    self.logger.error("Camera capture failed")
                cap.release()
            else:
                result = "❌ Camera test failed - Could not open camera"
                self.logger.error("Camera open failed")
                
        except Exception as e:
            result = f"❌ Camera test error: {str(e)}"
            self.logger.error(f"Camera test exception: {e}")
        
        self.result_label.setText(result)
        self.status_label.setText("System Status: Ready")
    
    def test_packages(self):
        """Test package imports"""
        self.logger.info("Testing package imports...")
        self.status_label.setText("System Status: Testing Packages...")
        
        results = []
        
        # Test packages
        packages_to_test = [
            ('OpenCV', 'cv2', lambda: cv2.__version__),
            ('NumPy', 'numpy', lambda: np.__version__),
            ('PyQt5', 'PyQt5.QtCore', lambda: 'Available'),
            ('PyTorch', 'torch', self._test_torch),
            ('TorchVision', 'torchvision', self._test_torchvision),
            ('PIL/Pillow', 'PIL', self._test_pil),
            ('PyZbar', 'pyzbar', self._test_pyzbar),
        ]
        
        for name, module, version_func in packages_to_test:
            try:
                __import__(module)
                version = version_func()
                results.append(f"✅ {name}: {version}")
                self.logger.info(f"{name} test passed")
            except Exception as e:
                results.append(f"❌ {name}: {str(e)}")
                self.logger.error(f"{name} test failed: {e}")
        
        result_text = "Package Test Results:\n\n" + "\n".join(results)
        self.result_label.setText(result_text)
        self.status_label.setText("System Status: Ready")
    
    def _test_torch(self):
        """Test PyTorch specifically"""
        import torch
        version = torch.__version__
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        return f"{version} ({device})"
    
    def _test_torchvision(self):
        """Test TorchVision specifically"""
        import torchvision
        return torchvision.__version__
    
    def _test_pil(self):
        """Test PIL/Pillow specifically"""
        from PIL import Image
        return Image.__version__ if hasattr(Image, '__version__') else 'Available'
    
    def _test_pyzbar(self):
        """Test PyZbar specifically"""
        try:
            import pyzbar
            return 'Available'
        except ImportError:
            return 'Not installed'


def main():
    """Main function to start the simple AI VDI System"""
    try:
        # Create Qt Application
        app = QApplication(sys.argv)
        
        # Create main window
        main_window = SimpleAIVDISystem()
        
        # Show main window
        main_window.show()
        
        print("AI VDI System Test Version started successfully!")
        print("Use the GUI buttons to test camera and package functionality.")
        
        # Run application
        sys.exit(app.exec_())
        
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()