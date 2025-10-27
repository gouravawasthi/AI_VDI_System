#!/usr/bin/env python3
"""
Reference Image & Mask Generator UI
Dedicated GUI application for creating and editing reference images and masks
for the AI VDI System.
"""

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                           QWidget, QPushButton, QLabel, QFrame, QComboBox,
                           QSlider, QSpinBox, QTextEdit, QFileDialog, QMessageBox,
                           QGroupBox, QGridLayout, QCheckBox, QProgressBar,
                           QTabWidget, QScrollArea, QSplitter, QButtonGroup,
                           QRadioButton, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont, QIcon
import json
from datetime import datetime

class CameraPreview(QLabel):
    """Custom QLabel for displaying camera feed with click events"""
    
    imageClicked = pyqtSignal(int, int)  # x, y coordinates
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #333; background-color: #000;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Camera Preview\nPress 'Start Camera' to begin")
        self.setScaledContents(True)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.imageClicked.emit(event.x(), event.y())

class MaskEditor(QLabel):
    """Custom QLabel for editing masks with drawing functionality"""
    
    maskChanged = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #333;")
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)
        
        # Drawing state
        self.drawing = False
        self.brush_size = 20
        self.brush_mode = True  # True for white (inspect), False for black (ignore)
        self.last_point = None
        
        # Images
        self.reference_image = None
        self.mask_image = None
        self.display_image = None
        
    def set_reference_image(self, image):
        """Set the reference image for mask editing"""
        self.reference_image = image.copy()
        h, w = image.shape[:2]
        
        # Initialize mask if not exists
        if self.mask_image is None:
            self.mask_image = np.zeros((h, w), dtype=np.uint8)
        else:
            self.mask_image = cv2.resize(self.mask_image, (w, h))
            
        self.update_display()
        
    def set_mask_image(self, mask):
        """Set the mask image"""
        self.mask_image = mask.copy()
        self.update_display()
        
    def update_display(self):
        """Update the display with reference image and mask overlay"""
        if self.reference_image is None:
            return
            
        # Create overlay
        overlay = self.reference_image.copy()
        if self.mask_image is not None:
            # Create colored overlay for mask
            mask_colored = cv2.applyColorMap(self.mask_image, cv2.COLORMAP_JET)
            mask_colored = cv2.bitwise_and(mask_colored, mask_colored, mask=self.mask_image)
            
            # Blend with reference image
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
            
        # Convert to QPixmap and display
        h, w, ch = overlay.shape
        bytes_per_line = ch * w
        qt_image = QImage(overlay.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        self.display_image = overlay
        self.setPixmap(QPixmap.fromImage(qt_image))
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.mask_image is not None:
            self.drawing = True
            self.last_point = (event.x(), event.y())
            self.draw_at_point(event.x(), event.y())
            
    def mouseMoveEvent(self, event):
        if self.drawing and self.mask_image is not None:
            self.draw_at_point(event.x(), event.y())
            self.last_point = (event.x(), event.y())
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.maskChanged.emit()
            
    def draw_at_point(self, x, y):
        """Draw on mask at given point"""
        if self.mask_image is None:
            return
            
        # Convert widget coordinates to image coordinates
        widget_size = self.size()
        image_h, image_w = self.mask_image.shape
        
        img_x = int((x / widget_size.width()) * image_w)
        img_y = int((y / widget_size.height()) * image_h)
        
        # Draw circle on mask
        color = 255 if self.brush_mode else 0
        cv2.circle(self.mask_image, (img_x, img_y), self.brush_size, color, -1)
        
        self.update_display()

class CameraThread(QThread):
    """Thread for handling camera operations"""
    
    frameReady = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.cap = None
        
    def start_camera(self):
        self.running = True
        self.start()
        
    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.quit()
        self.wait()
        
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.error.emit(f"Could not open camera {self.camera_index}")
            return
            
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frameReady.emit(frame)
            else:
                self.error.emit("Failed to read frame from camera")
                break
                
        if self.cap:
            self.cap.release()

class ReferenceImageUI(QMainWindow):
    """Main UI for Reference Image and Mask Generator"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI VDI System - Reference & Mask Generator")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data
        self.current_side = "front"
        self.sides = ["front", "back", "left", "right", "top", "bottom"]
        self.reference_images = {}
        self.mask_images = {}
        self.camera_thread = None
        self.current_frame = None
        
        # Setup directories
        self.setup_directories()
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        
        # Load existing data
        self.load_existing_data()
        
    def setup_directories(self):
        """Create necessary directories"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.ref_dir = os.path.join(self.base_dir, "data", "reference_images")
        self.mask_dir = os.path.join(self.base_dir, "data", "mask_images")
        
        os.makedirs(self.ref_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel - Image displays
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 3)
        
    def create_left_panel(self):
        """Create the left control panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Reference & Mask Generator")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Side selection
        side_group = QGroupBox("Select Side")
        side_layout = QVBoxLayout(side_group)
        
        self.side_combo = QComboBox()
        self.side_combo.addItems([side.title() for side in self.sides])
        side_layout.addWidget(self.side_combo)
        
        layout.addWidget(side_group)
        
        # Camera controls
        camera_group = QGroupBox("Camera Controls")
        camera_layout = QVBoxLayout(camera_group)
        
        self.start_camera_btn = QPushButton("Start Camera")
        self.stop_camera_btn = QPushButton("Stop Camera")
        self.capture_btn = QPushButton("Capture Reference")
        self.stop_camera_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        
        camera_layout.addWidget(self.start_camera_btn)
        camera_layout.addWidget(self.stop_camera_btn)
        camera_layout.addWidget(self.capture_btn)
        
        layout.addWidget(camera_group)
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        self.load_ref_btn = QPushButton("Load Reference Image")
        self.save_ref_btn = QPushButton("Save Reference Image")
        self.load_mask_btn = QPushButton("Load Mask")
        self.save_mask_btn = QPushButton("Save Mask")
        
        file_layout.addWidget(self.load_ref_btn)
        file_layout.addWidget(self.save_ref_btn)
        file_layout.addWidget(self.load_mask_btn)
        file_layout.addWidget(self.save_mask_btn)
        
        layout.addWidget(file_group)
        
        # Mask editing controls
        mask_group = QGroupBox("Mask Editing")
        mask_layout = QVBoxLayout(mask_group)
        
        # Brush mode
        mode_layout = QHBoxLayout()
        self.inspect_radio = QRadioButton("Inspect Area (White)")
        self.ignore_radio = QRadioButton("Ignore Area (Black)")
        self.inspect_radio.setChecked(True)
        mode_layout.addWidget(self.inspect_radio)
        mode_layout.addWidget(self.ignore_radio)
        mask_layout.addLayout(mode_layout)
        
        # Brush size
        brush_layout = QHBoxLayout()
        brush_layout.addWidget(QLabel("Brush Size:"))
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setRange(5, 100)
        self.brush_slider.setValue(20)
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(5, 100)
        self.brush_spin.setValue(20)
        brush_layout.addWidget(self.brush_slider)
        brush_layout.addWidget(self.brush_spin)
        mask_layout.addLayout(brush_layout)
        
        # Mask operations
        self.clear_mask_btn = QPushButton("Clear All (Ignore)")
        self.fill_mask_btn = QPushButton("Fill All (Inspect)")
        self.auto_edge_btn = QPushButton("Auto Edge Detection")
        
        mask_layout.addWidget(self.clear_mask_btn)
        mask_layout.addWidget(self.fill_mask_btn)
        mask_layout.addWidget(self.auto_edge_btn)
        
        layout.addWidget(mask_group)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_text = QTextEdit()
        self.progress_text.setMaximumHeight(150)
        self.progress_text.setReadOnly(True)
        progress_layout.addWidget(self.progress_text)
        
        layout.addWidget(progress_group)
        
        # Stretch
        layout.addStretch()
        
        return panel
        
    def create_right_panel(self):
        """Create the right image display panel"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Camera/Reference tab
        camera_tab = QWidget()
        camera_layout = QVBoxLayout(camera_tab)
        
        camera_label = QLabel("Camera / Reference Image")
        camera_label.setFont(QFont("Arial", 12, QFont.Bold))
        camera_layout.addWidget(camera_label)
        
        self.camera_preview = CameraPreview()
        camera_layout.addWidget(self.camera_preview)
        
        self.tab_widget.addTab(camera_tab, "Camera/Reference")
        
        # Mask editing tab
        mask_tab = QWidget()
        mask_layout = QVBoxLayout(mask_tab)
        
        mask_label = QLabel("Mask Editor")
        mask_label.setFont(QFont("Arial", 12, QFont.Bold))
        mask_layout.addWidget(mask_label)
        
        self.mask_editor = MaskEditor()
        mask_layout.addWidget(self.mask_editor)
        
        self.tab_widget.addTab(mask_tab, "Mask Editor")
        
        layout.addWidget(self.tab_widget)
        
        return panel
        
    def setup_connections(self):
        """Setup signal connections"""
        # Side selection
        self.side_combo.currentTextChanged.connect(self.on_side_changed)
        
        # Camera controls
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.capture_btn.clicked.connect(self.capture_reference)
        
        # File operations
        self.load_ref_btn.clicked.connect(self.load_reference_image)
        self.save_ref_btn.clicked.connect(self.save_reference_image)
        self.load_mask_btn.clicked.connect(self.load_mask)
        self.save_mask_btn.clicked.connect(self.save_mask)
        
        # Mask editing
        self.inspect_radio.toggled.connect(self.on_brush_mode_changed)
        self.brush_slider.valueChanged.connect(self.on_brush_size_changed)
        self.brush_spin.valueChanged.connect(self.on_brush_size_changed)
        self.clear_mask_btn.clicked.connect(self.clear_mask)
        self.fill_mask_btn.clicked.connect(self.fill_mask)
        self.auto_edge_btn.clicked.connect(self.auto_edge_detection)
        
        # Mask editor
        self.mask_editor.maskChanged.connect(self.on_mask_changed)
        
    def log_message(self, message):
        """Add message to progress log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.progress_text.append(f"[{timestamp}] {message}")
        
    def on_side_changed(self, side_text):
        """Handle side selection change"""
        self.current_side = side_text.lower()
        self.load_current_side_data()
        self.log_message(f"Switched to {side_text} side")
        
    def load_current_side_data(self):
        """Load reference and mask for current side"""
        side = self.current_side
        
        # Load reference image if exists
        ref_path = os.path.join(self.ref_dir, f"{side}_reference.jpg")
        if os.path.exists(ref_path):
            image = cv2.imread(ref_path)
            if image is not None:
                self.reference_images[side] = image
                self.display_reference_image(image)
                self.mask_editor.set_reference_image(image)
                self.log_message(f"Loaded existing reference for {side}")
                
        # Load mask if exists
        mask_path = os.path.join(self.mask_dir, f"{side}_mask.jpg")
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                self.mask_images[side] = mask
                self.mask_editor.set_mask_image(mask)
                self.log_message(f"Loaded existing mask for {side}")
                
    def display_reference_image(self, image):
        """Display reference image in camera preview"""
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.camera_preview.setPixmap(QPixmap.fromImage(qt_image))
        
    def start_camera(self):
        """Start camera feed"""
        if self.camera_thread is None:
            self.camera_thread = CameraThread()
            self.camera_thread.frameReady.connect(self.on_frame_ready)
            self.camera_thread.error.connect(self.on_camera_error)
            
        self.camera_thread.start_camera()
        self.start_camera_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)
        self.log_message("Camera started")
        
    def stop_camera(self):
        """Stop camera feed"""
        if self.camera_thread:
            self.camera_thread.stop_camera()
            
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.log_message("Camera stopped")
        
    def on_frame_ready(self, frame):
        """Handle new camera frame"""
        self.current_frame = frame.copy()
        self.display_reference_image(frame)
        
    def on_camera_error(self, error_msg):
        """Handle camera error"""
        QMessageBox.warning(self, "Camera Error", error_msg)
        self.log_message(f"Camera error: {error_msg}")
        
    def capture_reference(self):
        """Capture current frame as reference"""
        if self.current_frame is not None:
            self.reference_images[self.current_side] = self.current_frame.copy()
            self.mask_editor.set_reference_image(self.current_frame)
            
            # Create default mask
            h, w = self.current_frame.shape[:2]
            default_mask = np.ones((h, w), dtype=np.uint8) * 255  # All inspect
            self.mask_images[self.current_side] = default_mask
            self.mask_editor.set_mask_image(default_mask)
            
            self.log_message(f"Captured reference for {self.current_side}")
            self.tab_widget.setCurrentIndex(1)  # Switch to mask editor
            
    def load_reference_image(self):
        """Load reference image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Reference Image", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.reference_images[self.current_side] = image
                self.display_reference_image(image)
                self.mask_editor.set_reference_image(image)
                
                # Create default mask if not exists
                if self.current_side not in self.mask_images:
                    h, w = image.shape[:2]
                    default_mask = np.ones((h, w), dtype=np.uint8) * 255
                    self.mask_images[self.current_side] = default_mask
                    self.mask_editor.set_mask_image(default_mask)
                    
                self.log_message(f"Loaded reference from file for {self.current_side}")
                
    def save_reference_image(self):
        """Save current reference image"""
        if self.current_side in self.reference_images:
            file_path = os.path.join(self.ref_dir, f"{self.current_side}_reference.jpg")
            cv2.imwrite(file_path, self.reference_images[self.current_side])
            self.log_message(f"Saved reference for {self.current_side}")
        else:
            QMessageBox.warning(self, "Warning", "No reference image to save")
            
    def load_mask(self):
        """Load mask from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Mask", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if file_path:
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                self.mask_images[self.current_side] = mask
                self.mask_editor.set_mask_image(mask)
                self.log_message(f"Loaded mask from file for {self.current_side}")
                
    def save_mask(self):
        """Save current mask"""
        if self.current_side in self.mask_images:
            file_path = os.path.join(self.mask_dir, f"{self.current_side}_mask.jpg")
            cv2.imwrite(file_path, self.mask_images[self.current_side])
            self.log_message(f"Saved mask for {self.current_side}")
        else:
            QMessageBox.warning(self, "Warning", "No mask to save")
            
    def on_brush_mode_changed(self):
        """Handle brush mode change"""
        self.mask_editor.brush_mode = self.inspect_radio.isChecked()
        
    def on_brush_size_changed(self, value):
        """Handle brush size change"""
        self.mask_editor.brush_size = value
        self.brush_slider.setValue(value)
        self.brush_spin.setValue(value)
        
    def clear_mask(self):
        """Clear mask (set all to ignore)"""
        if self.current_side in self.reference_images:
            h, w = self.reference_images[self.current_side].shape[:2]
            self.mask_images[self.current_side] = np.zeros((h, w), dtype=np.uint8)
            self.mask_editor.set_mask_image(self.mask_images[self.current_side])
            self.log_message(f"Cleared mask for {self.current_side}")
            
    def fill_mask(self):
        """Fill mask (set all to inspect)"""
        if self.current_side in self.reference_images:
            h, w = self.reference_images[self.current_side].shape[:2]
            self.mask_images[self.current_side] = np.ones((h, w), dtype=np.uint8) * 255
            self.mask_editor.set_mask_image(self.mask_images[self.current_side])
            self.log_message(f"Filled mask for {self.current_side}")
            
    def auto_edge_detection(self):
        """Automatic edge detection for mask"""
        if self.current_side in self.reference_images:
            image = self.reference_images[self.current_side]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate edges to create mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(edges, kernel, iterations=2)
            
            self.mask_images[self.current_side] = mask
            self.mask_editor.set_mask_image(mask)
            self.log_message(f"Applied auto edge detection for {self.current_side}")
            
    def on_mask_changed(self):
        """Handle mask change"""
        if hasattr(self.mask_editor, 'mask_image') and self.mask_editor.mask_image is not None:
            self.mask_images[self.current_side] = self.mask_editor.mask_image.copy()
            
    def load_existing_data(self):
        """Load existing reference images and masks"""
        for side in self.sides:
            # Load reference
            ref_path = os.path.join(self.ref_dir, f"{side}_reference.jpg")
            if os.path.exists(ref_path):
                image = cv2.imread(ref_path)
                if image is not None:
                    self.reference_images[side] = image
                    
            # Load mask
            mask_path = os.path.join(self.mask_dir, f"{side}_mask.jpg")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    self.mask_images[side] = mask
                    
        # Update UI for current side
        self.load_current_side_data()
        
        # Log status
        ref_count = len(self.reference_images)
        mask_count = len(self.mask_images)
        self.log_message(f"Loaded {ref_count} reference images and {mask_count} masks")
        
    def closeEvent(self, event):
        """Handle application close"""
        if self.camera_thread:
            self.camera_thread.stop_camera()
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = ReferenceImageUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()