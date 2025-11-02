"""
Advanced Inspection Window - With image comparison and gradient edge detection
"""

import sys
import os
import csv
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QDialog,
                            QPushButton, QLabel, QFrame, QSpacerItem, QSizePolicy,
                            QGroupBox, QSlider, QCheckBox, QLineEdit, QTextEdit,
                            QProgressBar, QMessageBox, QComboBox, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QFont, QImage
import threading


class ImageProcessor:
    """Class for handling image processing operations"""
    
    @staticmethod
    def compute_gradient(image):
        """Compute gradient magnitude using Sobel operators"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))
        
        return gradient_magnitude
    
    @staticmethod
    def create_difference_overlay(current_grad, reference_grad, mask=None):
        """Create overlay highlighting differences between gradient images"""
        # Ensure both images are the same size
        if current_grad.shape != reference_grad.shape:
            reference_grad = cv2.resize(reference_grad, (current_grad.shape[1], current_grad.shape[0]))
        
        # Compute absolute difference
        diff = cv2.absdiff(current_grad, reference_grad)
        
        # Apply mask if provided
        if mask is not None:
            if mask.shape != diff.shape:
                mask = cv2.resize(mask, (diff.shape[1], diff.shape[0]))
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            diff = cv2.bitwise_and(diff, diff, mask=mask)
        
        # Create colored overlay
        # Red channel for differences
        overlay = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
        overlay[:, :, 2] = diff  # Red channel for differences
        overlay[:, :, 1] = reference_grad // 3  # Green channel for reference (dimmed)
        overlay[:, :, 0] = current_grad // 3   # Blue channel for current (dimmed)
        
        return overlay, diff
    
    @staticmethod
    def concatenate_frames(original, processed):
        """Concatenate original and processed frames horizontally"""
        # Ensure both images have the same height
        h1, w1 = original.shape[:2]
        h2, w2 = processed.shape[:2]
        
        target_height = min(h1, h2)
        
        # Resize if needed
        if h1 != target_height:
            original = cv2.resize(original, (int(w1 * target_height / h1), target_height))
        if h2 != target_height:
            processed = cv2.resize(processed, (int(w2 * target_height / h2), target_height))
        
        # Concatenate horizontally
        concatenated = np.hstack((original, processed))
        return concatenated


class CameraThread(QThread):
    """Thread for handling camera operations with image processing"""
    frame_ready = pyqtSignal(np.ndarray)
    processed_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)  # original, processed, concatenated
    
    def __init__(self,parent_window):
        super().__init__()
        self.parent = parent_window
        self.camera = None
        self.camera_index = 0  # Default camera device
        self.running = False
        self.flip_horizontal = False
        self.flip_vertical = False
        self.exposure = 0
        self.white_balance = 5000
        self.current_side = "Front"
        self.reference_images = {}
        self.mask_images = {}
        self.processor = ImageProcessor()
        self.load_reference_images()
        
    def load_reference_images(self):
        """Load reference images and masks for each side"""
        try:
            # Get reference images directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            ref_dir = os.path.join(project_root, "data", "reference_images")
            mask_dir = os.path.join(project_root, "data", "mask_images")
            
            # Create directories if they don't exist
            os.makedirs(ref_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            sides = ["Front", "Back", "Left", "Right"]
            
            for side in sides:
                # Load reference image
                ref_path = os.path.join(ref_dir, f"{side.lower()}_reference.jpg")
                mask_path = os.path.join(mask_dir, f"{side.lower()}_mask.jpg")
                
                if os.path.exists(ref_path):
                    ref_img = cv2.imread(ref_path)
                    if ref_img is not None:
                        self.reference_images[side] = ref_img
                        print(f"Loaded reference image for {side}")
                else:
                    # Create a placeholder reference image
                    placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 128
                    cv2.putText(placeholder, f"{side} Reference", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    self.reference_images[side] = placeholder
                    print(f"Created placeholder reference for {side}")
                
                if os.path.exists(mask_path):
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_img is not None:
                        self.mask_images[side] = mask_img
                        print(f"Loaded mask image for {side}")
                else:
                    # Create a placeholder mask (full white = inspect entire image)
                    mask_placeholder = np.ones((480, 640), dtype=np.uint8) * 255
                    self.mask_images[side] = mask_placeholder
                    print(f"Created placeholder mask for {side}")
                    
        except Exception as e:
            print(f"Error loading reference images: {e}")
    
    def set_current_side(self, side):
        """Set the current inspection side"""
        self.current_side = side
    
    def start_camera(self, camera_id=None):
        """Start camera capture"""
        try:
            # Use specified camera_id or fall back to self.camera_index
            if camera_id is None:
                camera_id = self.camera_index
                
            self.camera = cv2.VideoCapture(camera_id)
            if self.camera and self.camera.isOpened():
                # Set camera properties
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                
                self.running = True
                self.start()
                print(f"Camera {camera_id} started successfully")
                return True
            else:
                if self.camera:
                    self.camera.release()
                print(f"Failed to open camera {camera_id}")
                return False
        except Exception as e:
            print(f"Camera start error: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.running = False
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
            self.camera = None
        if self.isRunning():
            self.quit()
            self.wait(3000)
    
    def update_settings(self, flip_h=False, flip_v=False, exposure=0, wb=5000):
        """Update camera settings"""
        self.flip_horizontal = flip_h
        self.flip_vertical = flip_v
        self.exposure = exposure
        self.white_balance = wb
        
        if self.camera:
            try:
                self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            except:
                pass
    
    def run(self):
        """Main camera loop with image processing"""
        while self.running and self.camera and self.camera.isOpened():
            try:
                # If inspection is currently showing processed frames, skip capturing new video
                if self.parent.show_processed:
                    self.msleep(100)
                    continue

                ret, frame = self.camera.read()
                if not ret:
                    break

                # Apply transformations
                if self.flip_horizontal and self.flip_vertical:
                    frame = cv2.flip(frame, -1)
                elif self.flip_horizontal:
                    frame = cv2.flip(frame, 1)
                elif self.flip_vertical:
                    frame = cv2.flip(frame, 0)

                # Process frame for guiding display (green outline)
                processed_frame, concatenated_frame = self.process_frame(frame)
                self.processed_ready.emit(processed_frame, processed_frame, concatenated_frame)

            except Exception as e:
                print(f"Camera processing error: {e}")
                break

            self.msleep(33)
    
    def process_frame(self, frame):
        """
        Handles frame processing during inspection.
        When self.parent.show_processed == False → guiding outline + waiting text.
        When self.parent.show_processed == True  → analyze captured frame (once).
        """
        try:
            side = self.current_side
            gold_img = self.reference_images.get(side)
            gold_mask = self.mask_images.get(side)

            # No reference available
            if gold_img is None or gold_mask is None:
                guiding = np.zeros_like(frame)
                cv2.putText(guiding, "No reference found", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return guiding, self.processor.concatenate_frames(frame, guiding)

            # Resize references
            h, w = frame.shape[:2]
            gold_img = cv2.resize(gold_img, (w, h))
            gold_mask = cv2.resize(gold_mask, (w, h))

            # CASE 1: Pre-capture → show guiding outline
            if not self.parent.show_processed:
                guiding = frame.copy()
                contours, _ = cv2.findContours(gold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(guiding, contours, -1, (0, 255, 0), 2)

                waiting = np.zeros_like(frame)
                cv2.putText(waiting, "Waiting for capture...", (50, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                concat = self.processor.concatenate_frames(waiting, guiding)
                return guiding, concat

            # CASE 2: Post-capture → use analysis overlay
            else:
                from src.inspection.goldvsref import inspect_image
                result = inspect_image(gold_img, gold_mask, frame, f"Align {side}")
                if result["Status"] == 0:
                    err = np.zeros_like(frame)
                    cv2.putText(err, "Alignment failed", (40, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    concat = self.processor.concatenate_frames(err, frame)
                    return err, concat
                overlay = result["OutputImage"]
                concat = self.processor.concatenate_frames(frame, overlay)
                return overlay, concat

        except Exception as e:
            print(f"[Error] Frame processing failed: {e}")
            return frame, self.processor.concatenate_frames(frame, frame)
class AdvancedInspectionWindow(QWidget):
    """Advanced inspection window with image comparison"""
    
    inspection_complete = pyqtSignal(dict)
    window_closed = pyqtSignal()
   
    def __init__(self, parent=None):
        super().__init__()
        self.camera_active = False
        self.show_processed = False 
       
    
    # 🌟 Ask for station number at launch
        from PyQt5.QtWidgets import QInputDialog
        station_number, ok = QInputDialog.getInt(
            self,
            "Station Setup",
            "Enter station number:",
            value=1,
            min=1,
            max=50
        )
        if ok:
            self.station = station_number
        else:
            self.station = 1  # default
        print(f"✅ Station number set to: {self.station}")
        self.parent_window = parent
        self.barcode = ""
        self.current_side = 0
        self.inspection_sides = ["Front", "Back", "Left", "Right"]
        self.inspection_results = {}
        self.inspection_start_time = None
        self.side_start_time = None
        self.captured_frame = None
        self.current_camera_device = 0  # Single camera device for all sides
        self.camera_thread = CameraThread(self)
        self.init_ui()
        self.setup_camera()
        
    def init_ui(self):
        """Initialize the inspection interface"""
        self.setWindowTitle("AI VDI System - Advanced Inspection")
        self.showFullScreen()
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 25px;
                font-size: 16px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QPushButton#stopButton {
                background-color: #f44336;
            }
            QPushButton#stopButton:hover {
                background-color: #da190b;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 15px;
                padding-top: 15px;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
            }
            QLineEdit {
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 6px;
                font-size: 16px;
            }
            QLabel {
                font-size: 14px;
            }
        """)
        
        # Main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left panel - Controls
        self.create_control_panel(main_layout)
        
        # Center panel - Camera feed with comparison
        self.create_camera_panel(main_layout)
        
        # Right panel - Inspection progress and results
        self.create_inspection_panel(main_layout)
    
    def create_control_panel(self, main_layout):
        """Create the control panel on the left"""
        control_panel = QFrame()
        control_panel.setFixedWidth(350)
        control_panel.setStyleSheet("QFrame { border: 2px solid #ccc; border-radius: 10px; background-color: white; }")
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        
        # Title
        title = QLabel("Control Panel")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 15px;")
        control_layout.addWidget(title)
        
        # Barcode section
        self.create_barcode_section(control_layout)
        
        # Camera settings section
        self.create_camera_settings(control_layout)
        
        # Inspection controls
        self.create_inspection_controls(control_layout)
        
        # Add spacer
        control_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        main_layout.addWidget(control_panel)
    
    def create_barcode_section(self, layout):
        """Create barcode input section"""
        barcode_group = QGroupBox("Barcode Input")
        barcode_layout = QVBoxLayout()
        barcode_group.setLayout(barcode_layout)
        
        # Manual barcode input
        self.barcode_input = QLineEdit()
        self.barcode_input.setPlaceholderText("Enter barcode manually")
        barcode_layout.addWidget(self.barcode_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.scan_qr_button = QPushButton("Scan QR")
        self.scan_qr_button.clicked.connect(self.scan_qr_code)
        button_layout.addWidget(self.scan_qr_button)
        
        self.submit_barcode_button = QPushButton("Submit")
        self.submit_barcode_button.clicked.connect(self.submit_barcode)
        button_layout.addWidget(self.submit_barcode_button)
        
        barcode_layout.addLayout(button_layout)
        
        # Barcode display
        self.barcode_display = QLabel("No barcode entered")
        self.barcode_display.setStyleSheet("background-color: #f8f9fa; padding: 8px; border: 2px solid #ddd; border-radius: 5px;")
        barcode_layout.addWidget(self.barcode_display)
        
        layout.addWidget(barcode_group)
    
    def create_camera_settings(self, layout):
        """Create camera settings section"""
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QVBoxLayout()
        camera_group.setLayout(camera_layout)
        
        # Camera device selection
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Camera Device:"))
        self.camera_device_combo = QComboBox()
        self.camera_device_combo.addItems(["Camera 0", "Camera 1", "Camera 2", "Camera 3"])
        self.camera_device_combo.currentIndexChanged.connect(self.on_camera_device_changed)
        device_layout.addWidget(self.camera_device_combo)
        camera_layout.addLayout(device_layout)
        
        # Camera status
        self.camera_connection_status = QLabel("Camera: Select device for live preview")
        self.camera_connection_status.setStyleSheet("color: #f39c12; font-weight: bold; padding: 5px;")
        camera_layout.addWidget(self.camera_connection_status)
        
        # Flip settings
        self.flip_horizontal = QCheckBox("Flip Horizontal")
        self.flip_horizontal.toggled.connect(self.update_camera_settings)
        camera_layout.addWidget(self.flip_horizontal)
        
        self.flip_vertical = QCheckBox("Flip Vertical")
        self.flip_vertical.toggled.connect(self.update_camera_settings)
        camera_layout.addWidget(self.flip_vertical)
        
        # Exposure
        exposure_label = QLabel("Exposure:")
        camera_layout.addWidget(exposure_label)
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setRange(-10, 10)
        self.exposure_slider.setValue(0)
        self.exposure_slider.valueChanged.connect(self.update_camera_settings)
        camera_layout.addWidget(self.exposure_slider)
        
        # White balance
        wb_label = QLabel("White Balance:")
        camera_layout.addWidget(wb_label)
        self.wb_slider = QSlider(Qt.Horizontal)
        self.wb_slider.setRange(2000, 8000)
        self.wb_slider.setValue(5000)
        self.wb_slider.valueChanged.connect(self.update_camera_settings)
        camera_layout.addWidget(self.wb_slider)
        
        layout.addWidget(camera_group)
    
    def create_inspection_controls(self, layout):
        """Create inspection control buttons"""
        control_group = QGroupBox("Inspection Controls")
        control_layout = QVBoxLayout()
        control_group.setLayout(control_layout)
        
        self.start_inspection_button = QPushButton("Start Inspection")
        self.start_inspection_button.clicked.connect(self.start_inspection)
        self.start_inspection_button.setEnabled(False)
        control_layout.addWidget(self.start_inspection_button)
        
        self.capture_button = QPushButton("Capture & Analyze")
        self.capture_button.clicked.connect(self.capture_and_analyze)
        self.capture_button.setEnabled(False)
        control_layout.addWidget(self.capture_button)
        
        self.next_side_button = QPushButton("Next Side")
        self.next_side_button.clicked.connect(self.next_side)
        self.next_side_button.setEnabled(False)
        control_layout.addWidget(self.next_side_button)
        
        self.manual_override_button = QPushButton("Manual Override")
        self.manual_override_button.clicked.connect(self.manual_override)
        self.manual_override_button.setEnabled(False)
        control_layout.addWidget(self.manual_override_button)
        
        self.stop_inspection_button = QPushButton("Stop Inspection")
        self.stop_inspection_button.setObjectName("stopButton")
        self.stop_inspection_button.clicked.connect(self.stop_inspection)
        self.stop_inspection_button.setEnabled(False)
        control_layout.addWidget(self.stop_inspection_button)
        
        # Back to main menu button
        self.back_button = QPushButton("Back to Main Menu")
        self.back_button.setObjectName("stopButton")
        self.back_button.clicked.connect(self.back_to_main)
        control_layout.addWidget(self.back_button)
        
        layout.addWidget(control_group)
    
    def create_camera_panel(self, main_layout):
        """Create camera display panel with comparison view"""
        camera_panel = QFrame()
        camera_panel.setStyleSheet("QFrame { border: 2px solid #ccc; border-radius: 10px; background-color: white; }")
        camera_layout = QVBoxLayout()
        camera_panel.setLayout(camera_layout)
        
        # Title
        camera_title = QLabel("Live Camera Feed with Edge Comparison")
        camera_title.setFont(QFont("Arial", 16, QFont.Bold))
        camera_title.setAlignment(Qt.AlignCenter)
        camera_title.setStyleSheet("color: #2c3e50; margin: 10px;")
        camera_layout.addWidget(camera_title)
        
        # Camera feed (concatenated view)
        self.camera_label = QLabel("Camera Feed\n\nWaiting for camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(1000, 500)
        self.camera_label.setStyleSheet("""
            background-color: #2c3e50; 
            color: white; 
            font-size: 18px; 
            border-radius: 8px;
            border: 2px solid #34495e;
        """)
        camera_layout.addWidget(self.camera_label)
        
        # Info labels
        info_layout = QHBoxLayout()
        
        self.left_info = QLabel("Original Camera Feed")
        self.left_info.setAlignment(Qt.AlignCenter)
        self.left_info.setStyleSheet("color: #3498db; font-weight: bold;")
        info_layout.addWidget(self.left_info)
        
        self.right_info = QLabel("Gradient Comparison with Reference")
        self.right_info.setAlignment(Qt.AlignCenter)
        self.right_info.setStyleSheet("color: #e74c3c; font-weight: bold;")
        info_layout.addWidget(self.right_info)
        
        camera_layout.addLayout(info_layout)
        
        # Camera status
        self.camera_status = QLabel("Camera: Disconnected")
        self.camera_status.setAlignment(Qt.AlignCenter)
        self.camera_status.setStyleSheet("color: #e74c3c; font-size: 14px; margin: 5px;")
        camera_layout.addWidget(self.camera_status)
        
        main_layout.addWidget(camera_panel)
    
    def create_inspection_panel(self, main_layout):
        """Create inspection progress and results panel"""
        inspection_panel = QFrame()
        inspection_panel.setFixedWidth(350)
        inspection_panel.setStyleSheet("QFrame { border: 2px solid #ccc; border-radius: 10px; background-color: white; }")
        inspection_layout = QVBoxLayout()
        inspection_panel.setLayout(inspection_layout)
        
        # Title
        title = QLabel("Inspection Progress")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 15px;")
        inspection_layout.addWidget(title)
        
        # Progress section
        self.create_progress_section(inspection_layout)
        
        # Results section
        self.create_results_section(inspection_layout)
        
        main_layout.addWidget(inspection_panel)
    
    def create_progress_section(self, layout):
        """Create inspection progress section"""
        progress_group = QGroupBox("Current Progress")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)
        
        # Current side
        self.current_side_label = QLabel("Current Side: Not Started")
        self.current_side_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.current_side_label.setStyleSheet("color: #2c3e50; margin: 5px;")
        progress_layout.addWidget(self.current_side_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(self.inspection_sides))
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # Side status
        self.side_status_layout = QVBoxLayout()
        for side in self.inspection_sides:
            side_label = QLabel(f"{side}: Pending")
            side_label.setStyleSheet("color: #666; padding: 3px; font-size: 12px;")
            self.side_status_layout.addWidget(side_label)
        progress_layout.addLayout(self.side_status_layout)
        
        layout.addWidget(progress_group)
    
    def create_results_section(self, layout):
        """Create results display section"""
        results_group = QGroupBox("Inspection Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        
        # Overall result
        self.overall_result = QLabel("Overall: Pending")
        self.overall_result.setFont(QFont("Arial", 14, QFont.Bold))
        self.overall_result.setStyleSheet("color: #2c3e50; margin: 5px;")
        results_layout.addWidget(self.overall_result)
        
        # Detailed results
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.results_layout = QVBoxLayout()
        scroll_widget.setLayout(self.results_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(150)
        results_layout.addWidget(scroll_area)
        
        # Time information
        self.time_info = QLabel("Time: 00:00")
        self.time_info.setStyleSheet("color: #7f8c8d; font-size: 12px; margin: 5px;")
        results_layout.addWidget(self.time_info)
        
        layout.addWidget(results_group)
    
    def setup_camera(self):
        """Setup camera connection"""
        try:
            # Initialize camera with default device
            self.camera_thread.camera_index = self.current_camera_device
            self.camera_thread.processed_ready.connect(self.update_processed_feed)
            
            # Start camera immediately to show live preview
            self.camera_thread.start_camera()
            
            self.camera_status.setText(f"Camera: Device {self.current_camera_device} Connected")
            self.camera_status.setStyleSheet("color: #27ae60; font-size: 14px; margin: 5px;")
            
            # Update connection status
            self.camera_connection_status.setText(f"Camera {self.current_camera_device}: Connected")
            self.camera_connection_status.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px;")
            
        except Exception as e:
            print(f"Camera setup error: {e}")
            self.camera_status.setText("Camera: Error")
            self.camera_status.setStyleSheet("color: #e74c3c; font-size: 14px; margin: 5px;")
            self.camera_connection_status.setText("Camera: Connection Failed")
            self.camera_connection_status.setStyleSheet("color: #e74c3c; font-weight: bold; padding: 5px;")
            self.camera_label.setText("Camera Error")
    
    @pyqtSlot(np.ndarray)
    def update_camera_feed(self, frame):
        """Display mask outline only after Start Inspection is pressed."""
        try:
            '''
            if not self.camera_active:
                # Camera not started yet → show black screen
                black = np.zeros((480, 640, 3), dtype=np.uint8)
                rgb_image = cv2.cvtColor(black, cv2.COLOR_BGR2RGB)
            else:
                # Camera active → draw mask outline
                mask = self.camera_thread.mask_images.get(self.current_side, None)
                if mask is not None:
                    if len(mask.shape) == 3:
                        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    overlay = frame.copy()
                    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                else:
                    overlay = frame'''
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to QPixmap
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Camera feed update error: {e}")

    
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def update_processed_feed(self, original, processed, concatenated):
        """Update camera feed with processed comparison"""
       
        try:
            if not self.camera_active or not self.show_processed:
                # Right panel shows waiting text
                waiting_frame = np.zeros_like(original)
                cv2.putText(waiting_frame, "Waiting for frame capture...",
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                combined = np.hstack((original, waiting_frame))
            else:
                combined = concatenated

            rgb_image = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)

        except Exception as e:
            print(f"Processed feed update error: {e}")
    
    def update_camera_settings(self):
        """Update camera settings based on UI controls"""
        self.camera_thread.update_settings(
            flip_h=self.flip_horizontal.isChecked(),
            flip_v=self.flip_vertical.isChecked(),
            exposure=self.exposure_slider.value(),
            wb=self.wb_slider.value()
        )
    
    def on_camera_device_changed(self, index):
        """Handle camera device selection change and restart camera with new device"""
        self.current_camera_device = index
        print(f"Camera device changed to: {index}")
        
        # Restart camera with new device
        if self.camera_thread:
            self.camera_thread.stop_camera()
            QTimer.singleShot(200, lambda: self.start_camera_with_device(index))
        
        # Update status
        self.camera_connection_status.setText(f"Camera {index}: Connecting...")
        self.camera_connection_status.setStyleSheet("color: #f39c12; font-weight: bold; padding: 5px;")
    
    def start_camera_with_device(self, device_index):
        """Start camera with specific device index"""
        if self.camera_thread:
            self.camera_thread.camera_index = device_index
            self.camera_thread.start_camera()
            print(f"Camera started with device {device_index}")
            
            # Update status
            self.camera_connection_status.setText(f"Camera {device_index}: Connected")
            self.camera_connection_status.setStyleSheet("color: #27ae60; font-weight: bold; padding: 5px;")
    
    def get_camera_for_side(self, side):
        """Get the camera device index for any side (now uses single camera)"""
        return self.current_camera_device
    
    def scan_qr_code(self):
        """Scan QR code from camera feed"""
        QMessageBox.information(self, "QR Scanner", 
                               "QR Code scanning will be implemented with camera integration.")
    
    def submit_barcode(self):
        """Submit barcode and validate with server"""
        barcode = self.barcode_input.text().strip()
        if not barcode:
            QMessageBox.warning(self, "Error", "Please enter a barcode.")
            return
        
        # Validate barcode with server
        if self.validate_barcode(barcode):
            self.barcode = barcode
            self.barcode_display.setText(f"Barcode: {barcode}")
            self.barcode_display.setStyleSheet("background-color: #d4edda; padding: 8px; border: 2px solid #c3e6cb; border-radius: 5px; color: #155724;")
            self.start_inspection_button.setEnabled(True)
            QMessageBox.information(self, "Success", "Barcode validated successfully!")
        
        '''else:
            self.barcode_display.setStyleSheet("background-color: #f8d7da; padding: 8px; border: 2px solid #f5c6cb; border-radius: 5px; color: #721c24;")
            QMessageBox.critical(self, "Validation Failed", 
                               "Barcode was rejected in previous process. Inspection cannot proceed.")'''
        
    def validate_barcode(self, barcode):
        """Validate barcode with Flask + SQLite backend"""
        import requests
        from logger import JSONLogger  # Local import to avoid circular dependency
        self.logger = JSONLogger(self.barcode,self.station) # Local import to avoid circular dependency
        try:
            api_url = f"http://127.0.0.1:5000/validate/{barcode}"
            response = requests.get(api_url, timeout=5)

            if response.status_code == 200:
                return True 
            elif response.status_code == 401:
                data = response.json()
                details = data.get("inspection_details", {})

                # Build detail string
                info_text = (
                    f"<b>Barcode:</b> {data['barcode_id']}<br>"
                    f"<b>Status:</b> {data['status']}<br><br>"
                    f"<b>Inspection Details:</b><br>"
                    f"Front: {details.get('front', '-')}, "
                    f"Back: {details.get('back', '-')}, "
                    f"Right: {details.get('right_side', '-')}, "
                    f"Left: {details.get('left_side', '-')}, "
                    f"Top: {details.get('top', '-')}, "
                    f"Down: {details.get('down', '-')}"
                )

                # Show custom dialog
                dialog = QDialog(self)
                dialog.setWindowTitle("Duplicate Barcode Found")
                layout = QVBoxLayout(dialog)
                label = QLabel(info_text)
                label.setTextFormat(1)  # Allow HTML
                layout.addWidget(label)

                # Buttons
                append_btn = QPushButton("Append")
                update_btn = QPushButton("Update")
                cancel_btn = QPushButton("Cancel")

                layout.addWidget(append_btn)
                layout.addWidget(update_btn)
                layout.addWidget(cancel_btn)

                # Define actions
                def append_action():
                    dialog.done(1)  # proceed forward
                def update_action():
                    dialog.done(2)  # re-inspection
                def cancel_action():
                    dialog.done(0)  # invalidate
                append_btn.clicked.connect(append_action)
                update_btn.clicked.connect(update_action)
                cancel_btn.clicked.connect(cancel_action)

                result = dialog.exec_()

                # Interpret user choice
                if result == 1:
                    print("User chose to append and continue.")
                    self.logger.set_type(1)
                    return True
                elif result == 2:
                    print("User chose to update previous data.")
                    self.logger.set_type(2)
                    return True            
                else:
                    print("User cancelled.")
                    return False
            elif response.status_code == 404:
                QMessageBox.warning(self, "Not Found", f"Barcode '{barcode}' not found in database.")
                return False
            elif response.status_code == 409:
                QMessageBox.critical(self, "Validation Failed", 
                                     f"Barcode '{barcode}' was rejected in previous process. Inspection cannot proceed.")
                return False
            else:
                QMessageBox.critical(self, "Error", f"Server error: {response.status_code}")
                return False
        except requests.ConnectionError:
            QMessageBox.critical(self, "Connection Error", "Could not connect to local API server.")
            return False
        
    '''def validate_barcode(self, barcode):
        """Validate barcode with server API"""
        try:
            # Mock validation
            return not barcode.endswith('0')
        except Exception as e:
            print(f"API validation error: {e}")
            return False'''
    
    def start_inspection(self):
        """Triggered when Start Inspection is clicked"""
        self.camera_active = True
        self.show_processed = False
        if not self.camera_thread.running:
            if not self.camera_thread.start_camera(0):
                QMessageBox.critical(self, "Camera Error", "Unable to start camera.")
                return
        """Start the inspection process"""
        
        self.inspection_start_time = datetime.now()
        self.current_side = 0
        self.inspection_results = {}
        
        # Enable controls
        self.start_inspection_button.setEnabled(False)
        self.capture_button.setEnabled(True)
        self.next_side_button.setEnabled(False)
        self.manual_override_button.setEnabled(True)
        self.stop_inspection_button.setEnabled(True)
        
        # Start first side
        self.start_side_inspection()
    
    def start_side_inspection(self):
        
        """Start inspection of current side"""
        if self.current_side < len(self.inspection_sides):
            side_name = self.inspection_sides[self.current_side]
            self.side_start_time = datetime.now()
            self.current_side_label.setText(f"Current Side: {side_name}")
            
            # Update camera processing for current side
            self.camera_thread.set_current_side(side_name)
            
            # Update side status
            for i in range(self.side_status_layout.count()):
                label = self.side_status_layout.itemAt(i).widget()
                if label and hasattr(label, 'setText'):
                    if i == self.current_side:
                        label.setText(f"{self.inspection_sides[i]}: In Progress")
                        label.setStyleSheet("color: #3498db; font-weight: bold; padding: 3px; font-size: 12px;")
    
    def capture_and_analyze(self):
        """Capture current frame and perform analysis"""
        self.show_processed = True

        # Capture the latest frame from camera
        ret, frame = self.camera_thread.camera.read()
        if not ret:
            QMessageBox.warning(self, "Camera Error", "Failed to capture frame.")
            return
        
        # Freeze frame
        self.captured_frame = frame.copy()
        side_name = self.inspection_sides[self.current_side]

        # Perform analysis on captured frame
        processed, concatenated = self.camera_thread.process_frame(self.captured_frame)
    
        # Force display of analyzed frame
        self.update_processed_feed(self.captured_frame, processed, concatenated)
        # Record result
        result = self.perform_side_inspection(side_name)
        side_time = (datetime.now() - self.side_start_time).total_seconds()
        self.inspection_results[side_name] = {
            'result': result,
            'time': side_time,
            'timestamp': datetime.now()
        }
        if result == "PASS":
            # Update UI
            self.update_side_status(self.current_side, result)
            self.logger.log_step(side_name, result)
            # Enable next side button
            self.capture_button.setEnabled(False)
            self.next_side_button.setEnabled(True)
            
            # Show result message
            msg = f"Side {side_name} analyzed!\n\nResult: {result}\n"
            QMessageBox.information(self, "Inspection Result", msg)
        if result == "FAIL":
            self.next_side_button.setEnabled(False)
            QMessageBox.warning(self, "Inspection Failed",
                                f"{side_name} failed inspection!\n\n"
                                "Attempting Manual Override...")
            # Trigger manual override popup
            self.handle_manual_override_popup(side_name)
    def handle_manual_override_popup(self, side_name):
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Manual Override Required")
        dialog.setText(f"{side_name} FAILED.\nApply override or send log?")
        override_btn = dialog.addButton("Override", QMessageBox.AcceptRole)
        fail_btn = dialog.addButton("Send Log", QMessageBox.RejectRole)
        dialog.exec_()
        if dialog.clickedButton() == override_btn:
            self.inspection_results[side_name]['result'] = 'PASS'
            self.logger.log_step(side_name,  self.inspection_results[side_name]['result'])
            self.update_side_status(self.current_side, "PASS")
            self.next_side()
        else:
            self.logger.set_final_status("FAIL")

            self.logger.log_step(side_name,  self.inspection_results[side_name]['result'])
            self.logger.send_log_on_complete("http://127.0.0.1:5000/receive_log")
            self.update_side_status(self.inspection_sides.index(side_name), "FAIL")
            self.next_side_button.setEnabled(False)
            self.capture_button.setEnabled(False)

    def next_side(self):
        """Move to next side of inspection"""
        self.show_processed = False
        self.captured_frame = None
        self.current_side += 1
        self.progress_bar.setValue(self.current_side)
        
        if self.current_side < len(self.inspection_sides):
            self.start_side_inspection()
            self.capture_button.setEnabled(True)
            self.next_side_button.setEnabled(False)
        else:
            self.complete_inspection()
    
    def perform_side_inspection(self, side_name):
        '''refrence_image = self.mask_images(side_name)
        mask_image = self.reference_images(side_name)
        from preprocess import FramePreprocessor
        """Perform actual inspection logic for a side"""
        preprocessor = FramePreprocessor(camera_index=0, duration=1)
        processed_frame = preprocessor.preprocess()

        if processed_frame is None:
            QMessageBox.warning(self, "Camera Error", "Failed to capture frame.")
            return'''
        ''' Implement actual inspection logic here
         return "PASS" or "FAIL" based on analysis '''
        
        import random
        return "PASS" if random.random() > 0.25 else "FAIL"
    
    def update_side_status(self, side_index, result):
        """Update the status of a specific side"""
        if side_index < self.side_status_layout.count():
            label = self.side_status_layout.itemAt(side_index).widget()
            if label:
                side_name = self.inspection_sides[side_index]
                label.setText(f"{side_name}: {result}")
                if result == "PASS":
                    label.setStyleSheet("color: #27ae60; font-weight: bold; padding: 3px; font-size: 12px;")
                else:
                    label.setStyleSheet("color: #e74c3c; font-weight: bold; padding: 3px; font-size: 12px;")
    
    def complete_inspection(self):
        """Complete the inspection process"""
        total_time = (datetime.now() - self.inspection_start_time).total_seconds()
        
        # Determine overall result
        failed_sides = [side for side, data in self.inspection_results.items() if data['result'] == 'FAIL']
        overall_result = "FAIL" if failed_sides else "PASS"
        
        # Update UI
        self.overall_result.setText(f"Overall: {overall_result}")
        if overall_result == "PASS":
            self.overall_result.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 14px; margin: 5px;")
        else:
            self.overall_result.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 14px; margin: 5px;")
        
        self.time_info.setText(f"Total Time: {total_time:.1f}s")
        
        # Display detailed results
        self.display_detailed_results(failed_sides)
        
        # Log results
        #self.log_inspection_results(overall_result, total_time, failed_sides)
        # ✅ Finalize JSON log
        self.logger.set_final_status(overall_result)
        self.logger.send_log_on_complete("http://127.0.0.1:5000/receive_log")

        # Disable controls
        self.capture_button.setEnabled(False)
        self.next_side_button.setEnabled(False)
        self.stop_inspection_button.setEnabled(False)
        self.start_inspection_button.setEnabled(True)
        
        # Show final result
        result_msg = f"Inspection Complete!\n\nResult: {overall_result}\nTotal Time: {total_time:.1f}s"
        if failed_sides:
            result_msg += f"\n\nFailed Sides: {', '.join(failed_sides)}"
        
        QMessageBox.information(self, "Inspection Complete", result_msg)
    
    def display_detailed_results(self, failed_sides):
        """Display detailed inspection results"""
        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)
        
        # Add detailed results
        for side, data in self.inspection_results.items():
            result_label = QLabel(f"{side}: {data['result']} ({data['time']:.1f}s)")
            if data['result'] == "PASS":
                result_label.setStyleSheet("color: #27ae60; padding: 2px; font-size: 11px;")
            else:
                result_label.setStyleSheet("color: #e74c3c; padding: 2px; font-size: 11px;")
            self.results_layout.addWidget(result_label)
    
    def log_inspection_results(self, overall_result, total_time, failed_sides):
        """Log inspection results to CSV file"""
        try:
            # Ensure logs directory exists
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            logs_dir = os.path.join(project_root, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # CSV file path
            csv_file = os.path.join(logs_dir, "advanced_inspection_log.csv")
            
            # Check if file exists to write header
            file_exists = os.path.exists(csv_file)
            
            with open(csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                
                # Write header if new file
                if not file_exists:
                    header = ['Timestamp', 'Barcode', 'Overall_Result', 'Total_Time', 'Failed_Sides']
                    for side in self.inspection_sides:
                        header.extend([f'{side}_Result', f'{side}_Time'])
                    writer.writerow(header)
                
                # Write inspection data
                row = [
                    datetime.now().isoformat(),
                    self.barcode,
                    overall_result,
                    f"{total_time:.2f}",
                    ';'.join(failed_sides) if failed_sides else 'None'
                ]
                
                # Add individual side results
                for side in self.inspection_sides:
                    if side in self.inspection_results:
                        row.extend([
                            self.inspection_results[side]['result'],
                            f"{self.inspection_results[side]['time']:.2f}"
                        ])
                    else:
                        row.extend(['N/A', '0'])
                
                writer.writerow(row)
                print(f"Advanced inspection results logged to: {csv_file}")
                
        except Exception as e:
            print(f"Error logging results: {e}")
    
    def manual_override(self):
        """Handle manual override"""
        if not self.inspection_results:
            QMessageBox.warning(self, "No Inspection", "No inspection has been performed yet.")
            return
            
        reply = QMessageBox.question(self, "Manual Override", 
                                   "Apply manual override to current inspection?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
             side=self.inspection_sides[self.current_side]
             self.logger.set_final_status("FAIL")
             self.logger.log_step(side, "FAIL")
             self.logger.send_log_on_complete("http://127.0.0.1:5000/receive_log")
             self.update_side_status(self.inspection_sides.index(side), "FAIL")
             self.next_side_button.setEnabled(False)
             self.capture_button.setEnabled(False)
             QMessageBox.information(self, "Override Applied", "Manual override has been applied.")
    
    def stop_inspection(self):
        
        """Stop the current inspection"""
        reply = QMessageBox.question(self, "Stop Inspection", 
                                   "Are you sure you want to stop the inspection?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.show_processed=False
            self.camera_active=False
            self.camera_thread.stop_camera()

            self.reset_inspection()
    
    def reset_inspection(self):
        """Reset inspection to initial state"""
        self.current_side = 0
        self.inspection_results = {}
        self.current_side_label.setText("Current Side: Not Started")
        self.progress_bar.setValue(0)
        self.overall_result.setText("Overall: Pending")
        self.time_info.setText("Time: 00:00")
        
        # Reset side status
        for i in range(self.side_status_layout.count()):
            label = self.side_status_layout.itemAt(i).widget()
            if label and hasattr(label, 'setText'):
                side_name = self.inspection_sides[i]
                label.setText(f"{side_name}: Pending")
                label.setStyleSheet("color: #666; padding: 3px; font-size: 12px;")
        
        # Reset buttons
        self.start_inspection_button.setEnabled(bool(self.barcode))
        self.capture_button.setEnabled(False)
        self.next_side_button.setEnabled(False)
        self.manual_override_button.setEnabled(False)
        self.stop_inspection_button.setEnabled(False)
    
    def back_to_main(self):
        """Return to main window"""
        reply = QMessageBox.question(self, "Back to Main", 
                                   "Return to main menu?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.window_closed.emit()
            self.close()
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.camera_thread.stop_camera()
        self.window_closed.emit()
        event.accept()


def main():
    """Main function for testing advanced inspection window"""
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = AdvancedInspectionWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()