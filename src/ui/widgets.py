"""
Custom widgets for AI VDI System
"""

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont
import cv2


class CameraFeedWidget(QWidget):
    """Widget to display camera feed"""
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.cap = None
        self.timer = QTimer()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the camera feed widget"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Camera display label
        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid gray;")
        layout.addWidget(self.camera_label)
    
    def start_camera(self):
        """Start camera feed"""
        # TODO: Initialize camera capture
        pass
    
    def stop_camera(self):
        """Stop camera feed"""
        # TODO: Release camera resources
        pass


class StatusPanelWidget(QWidget):
    """Widget to display system status and inspection results"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the status panel"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Status labels
        self.system_status = QLabel("System Status: Ready")
        self.inspection_result = QLabel("Last Inspection: None")
        self.barcode_result = QLabel("Last Barcode: None")
        
        # Style the labels
        font = QFont()
        font.setPointSize(12)
        for label in [self.system_status, self.inspection_result, self.barcode_result]:
            label.setFont(font)
            layout.addWidget(label)
    
    def update_system_status(self, status):
        """Update system status"""
        self.system_status.setText(f"System Status: {status}")
    
    def update_inspection_result(self, result):
        """Update inspection result"""
        self.inspection_result.setText(f"Last Inspection: {result}")
    
    def update_barcode_result(self, barcode):
        """Update barcode result"""
        self.barcode_result.setText(f"Last Barcode: {barcode}")


class ControlPanelWidget(QWidget):
    """Widget for system controls"""
    
    start_inspection = pyqtSignal()
    stop_inspection = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize the control panel"""
        layout = QHBoxLayout()
        self.setLayout(layout)
        
        # TODO: Add control buttons
        # TODO: Add configuration options