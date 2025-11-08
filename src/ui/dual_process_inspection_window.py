"""
Dual Process Inspection Window - Top & Bottom Component Detection
Separate API calls for each process
"""
from algo_top import  ImageInspectionPipeline
import sys
import os
import cv2
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QDialog,
                            QPushButton, QLabel, QFrame, QSpacerItem, QSizePolicy,
                            QGroupBox, QCheckBox, QLineEdit, QTextEdit,
                            QProgressBar, QMessageBox, QScrollArea, QApplication)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QFont, QImage
import requests
from src.ui.advanced_inspection_window import ImageProcessor
from algo_top import ImageInspectionPipeline
from algo_bottom import SobelBottom
from api_manager import APIManager



class CameraThread(QThread):
    """Thread for handling camera operations"""
    frame_ready = pyqtSignal(np.ndarray)
    processed_ready = pyqtSignal(np.ndarray, np.ndarray, np.ndarray) 
    
    def __init__(self,parent_window):
        super().__init__()
        self.top_detector = ImageInspectionPipeline()
        self.top_rois =self.top_rois = {
                'Antenna': (172, 112, 105, 64),
                'Capacitor': (561, 141, 194, 153),
                'Speaker': (803, 221, 165, 198),
            }
        self.bottom_rois = {
            'Plate': (830, 199, 346, 419),
                }
        self.top_ref=None
        self.bottom_ref=None
        self.bottom_detector =SobelBottom()
        self.parent = parent_window
        self.camera = None
        self.running = False
        self.flip_horizontal = False
        self.flip_vertical = False
        self.current_side= "Top"
        self.reference_images = {}
        self.mask_images = {}
        self.processor=ImageProcessor()
        self.load_reference_images()
        

    def load_reference_images(self):
        """Load reference images and masks for each side"""
        try:
            # Get reference images directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            ref_dir = os.path.join(project_root, "data", "INLINE")
            mask_dir = os.path.join(project_root, "data", "INLINE")
            
            # Create directories if they don't exist
            os.makedirs(ref_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)
            
            sides = ["Top", "Bottom"]
            
            for side in sides:
                # Load reference image
                ref_path = os.path.join(ref_dir, f"{side.lower()}_reference.png")
                mask_path = os.path.join(mask_dir, f"{side.lower()}_mask.png")
                
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
    def start_camera(self, camera_id=0):
        """Start camera capture"""
        try:
            self.camera = cv2.VideoCapture(camera_id)
            if self.camera and self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                self.running = True
                self.start()
                return True
            return False
        except Exception as e:
            print(f"Camera start error: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera thread"""
        self.running = False
        if self.isRunning():
            self.quit()
            self.wait(2000)
        if self.camera:
            try:
                self.camera.release()
            except Exception as e:
                print(f"Camera release error: {e}")
            self.camera = None
    
    def update_settings(self, flip_h=False, flip_v=False):
        """Update camera settings"""
        self.flip_horizontal = flip_h
        self.flip_vertical = flip_v
    def process_frame(self, frame):
        """
        Handles frame processing during inspection.
        When self.parent.show_processed == False → guiding outline + waiting text.
        When self.parent.show_processed == True  → analyze captured frame (once).
        Dynamically uses TOP or BOTTOM detectors based on current process.
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

            # Resize references to match frame
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

            # CASE 2: Post-capture → analyze using correct detector
            else:
                overlay = None
                concat = None

                # --- TOP Process ---
                if self.parent.current_process == "TOP" and self.top_detector:
                    try:
                        
                        result = self.top_detector.run(
                            ref_img=self.reference_images.get("Top"),
                            input_img=frame,
                            roi_definitions=self.top_rois
                        )
                        overlay = result["annotated"]
                        self.parent.top_results = result["results"]
                    except Exception as e:
                        print(f"[Error] TOP detector failed: {e}")
                        overlay = np.zeros_like(frame)
                        cv2.putText(overlay, "TOP detector error", (40, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # --- BOTTOM Process ---
                elif self.parent.current_process == "BOTTOM" and self.bottom_detector:
                    try:
                        result = self.bottom_detector.analyze_roi(
                            ref_img=self.reference_images.get("Bottom"),
                            inp_img=frame,
                            roi_definitions=self.bottom_rois
                        )
                        overlay = result["annotated"]
                        self.parent.bottom_results = result["results"]
                    except Exception as e:
                        print(f"[Error] BOTTOM detector failed: {e}")
                        overlay = np.zeros_like(frame)
                        cv2.putText(overlay, "BOTTOM detector error", (40, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                else:
                    # No valid detector
                    overlay = np.zeros_like(frame)
                    cv2.putText(overlay, "No active detector", (40, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Combine original + annotated
                concat = self.processor.concatenate_frames(frame, overlay)
                return overlay, concat

        except Exception as e:
            print(f"[Error] Frame processing failed: {e}")
            err = np.zeros_like(frame)
            cv2.putText(err, f"Processing Error: {e}", (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return err, self.processor.concatenate_frames(frame, err)

    def run(self):
        """Main camera loop"""
        while self.running and self.camera and self.camera.isOpened():
            try:
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
                
                processed_frame, concatenated_frame = self.process_frame(frame)
                self.processed_ready.emit(processed_frame, processed_frame, concatenated_frame)
            except Exception as e:
                print(f"Camera processing error: {e}")
                break
            
            self.msleep(33)


class DualProcessInspectionWindow(QWidget):
    """Dual process inspection window with separate API calls"""
    inspection_complete = pyqtSignal(dict)
    window_closed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__()
        self.camera_active = False
        self.show_processed = False 
       
        self.top_detector = None
        self.parent_window = parent
        self.barcode = ""
        self.station = 1
        
        self.api_manager = APIManager.create_workflow("INLINE_TOP_TO_INLINE_BOTTOM")
        self.logger = None
        
        # Process states
        self.current_process = None  # "TOP" or "BOTTOM"
        self.top_completed = False
        self.bottom_completed = False
        
        # Component results
        self.top_results = {}
        self.bottom_results = {}
        
        # Captured frames
        self.top_frame = None
        self.bottom_frame = None
        
        # Detectors (will be initialized with reference images)
        
        
        # Camera
        self.camera_thread = CameraThread(self)
        self.camera_active = False
        
        # Get station number
        from PyQt5.QtWidgets import QInputDialog
        station_number, ok = QInputDialog.getInt(
            self, "Station Setup", "Enter station number:",
            value=1, min=1, max=50
        )
        if ok:
            self.station = station_number
        
        print(f"✅ Station number set to: {self.station}")
        
        self.init_ui()
        self.load_reference_images()
        self.setup_camera()
    
    def init_ui(self):
        """Initialize the inspection interface"""
        self.setWindowTitle("Dual Process Component Inspection")
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
        """)
        
        # Main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left panel - Controls
        self.create_control_panel(main_layout)
        
        # Center panel - Camera feed
        self.create_camera_panel(main_layout)
        
        # Right panel - Results
        self.create_results_panel(main_layout)
    
    def create_control_panel(self, main_layout):
        """Create control panel"""
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
        
        # Camera settings
        self.create_camera_settings(control_layout)
        
        # Process controls
        self.create_process_controls(control_layout)
        
        control_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        main_layout.addWidget(control_panel)
    
    def create_barcode_section(self, layout):
        """Create barcode input section"""
        barcode_group = QGroupBox("Barcode Input")
        barcode_layout = QVBoxLayout()
        barcode_group.setLayout(barcode_layout)
        
        self.barcode_input = QLineEdit()
        self.barcode_input.setPlaceholderText("Enter barcode")
        barcode_layout.addWidget(self.barcode_input)
        
        button_layout = QHBoxLayout()
        
        self.scan_qr_button = QPushButton("Scan QR")
        self.scan_qr_button.clicked.connect(self.scan_qr_code)
        button_layout.addWidget(self.scan_qr_button)
        
        self.submit_barcode_button = QPushButton("Submit")
        self.submit_barcode_button.clicked.connect(self.submit_barcode)
        button_layout.addWidget(self.submit_barcode_button)
        
        barcode_layout.addLayout(button_layout)
        
        self.barcode_display = QLabel("No barcode entered")
        self.barcode_display.setStyleSheet("background-color: #f8f9fa; padding: 8px; border: 2px solid #ddd; border-radius: 5px;")
        barcode_layout.addWidget(self.barcode_display)
        
        layout.addWidget(barcode_group)
    
    def create_camera_settings(self, layout):
        """Create camera settings"""
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QVBoxLayout()
        camera_group.setLayout(camera_layout)
        
        self.flip_horizontal = QCheckBox("Flip Horizontal")
        self.flip_horizontal.toggled.connect(self.update_camera_settings)
        camera_layout.addWidget(self.flip_horizontal)
        
        self.flip_vertical = QCheckBox("Flip Vertical")
        self.flip_vertical.toggled.connect(self.update_camera_settings)
        camera_layout.addWidget(self.flip_vertical)
        
        layout.addWidget(camera_group)
    
    def create_process_controls(self, layout):
        """Create process control buttons"""
        control_group = QGroupBox("Process Controls")
        control_layout = QVBoxLayout()
        control_group.setLayout(control_layout)
        
        # Process 1 - TOP
        self.start_top_button = QPushButton("Start BOTTOM Process")
        self.start_top_button.clicked.connect(self.start_top_process)
        self.start_top_button.setEnabled(False)
        control_layout.addWidget(self.start_top_button)
        
        self.capture_top_button = QPushButton("Capture BOTTOM")
        self.capture_top_button.clicked.connect(self.capture_top)
        self.capture_top_button.setEnabled(False)
        control_layout.addWidget(self.capture_top_button)
      
        self.submit_top_button = QPushButton("Submit BOTTOM")
        self.submit_top_button.clicked.connect(self.submit_top_to_api)
        self.submit_top_button.setEnabled(False)
        self.submit_top_button.setStyleSheet("background-color: #FF9800;")
        control_layout.addWidget(self.submit_top_button)
        
        # Separator
        separator = QLabel("─" * 40)
        separator.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(separator)
        
        # Process 2 - BOTTOM
        self.start_bottom_button = QPushButton("Start TOP Process")
        self.start_bottom_button.clicked.connect(self.start_bottom_process)
        self.start_bottom_button.setEnabled(False)
        control_layout.addWidget(self.start_bottom_button)
        
        self.capture_bottom_button = QPushButton("Capture TOP")
        self.capture_bottom_button.clicked.connect(self.capture_bottom)
        self.capture_bottom_button.setEnabled(False)
        control_layout.addWidget(self.capture_bottom_button)
        
        self.submit_bottom_button = QPushButton("Submit TOP")
        self.submit_bottom_button.clicked.connect(self.submit_bottom_to_api)
        self.submit_bottom_button.setEnabled(False)
        self.submit_bottom_button.setStyleSheet("background-color: #FF9800;")
        control_layout.addWidget(self.submit_bottom_button)
        self.stop_inspection_button = QPushButton("Stop Inspection")
        self.stop_inspection_button.setObjectName("stopButton")
        self.stop_inspection_button.clicked.connect(self.stop_inspection)
        control_layout.addWidget(self.stop_inspection_button)
        # Back button
        self.back_button = QPushButton("Back to Main Menu")
        self.back_button.setObjectName("stopButton")
        self.back_button.clicked.connect(self.back_to_main)
        control_layout.addWidget(self.back_button)
        
        layout.addWidget(control_group)
    
    def create_camera_panel(self, main_layout):
        """Create camera display panel"""
        camera_panel = QFrame()
        camera_panel.setStyleSheet("QFrame { border: 2px solid #ccc; border-radius: 10px; background-color: white; }")
        camera_layout = QVBoxLayout()
        camera_panel.setLayout(camera_layout)
        
        camera_title = QLabel("Live Camera Feed")
        camera_title.setFont(QFont("Arial", 16, QFont.Bold))
        camera_title.setAlignment(Qt.AlignCenter)
        camera_title.setStyleSheet("color: #2c3e50; margin: 10px;")
        camera_layout.addWidget(camera_title)
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
        
        '''
        self.camera_label = QLabel("Camera Feed\n\nWaiting for camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(800, 600)
        self.camera_label.setStyleSheet("""
            background-color: #2c3e50; 
            color: white; 
            font-size: 18px; 
            border-radius: 8px;
        """)
        camera_layout.addWidget(self.camera_label)'''
        
        self.camera_status = QLabel("Camera: Disconnected")
        self.camera_status.setAlignment(Qt.AlignCenter)
        self.camera_status.setStyleSheet("color: #e74c3c; font-size: 14px; margin: 5px;")
        camera_layout.addWidget(self.camera_status)
        
        main_layout.addWidget(camera_panel)
    
    def create_results_panel(self, main_layout):
        """Create results display panel"""
        results_panel = QFrame()
        results_panel.setFixedWidth(350)
        results_panel.setStyleSheet("QFrame { border: 2px solid #ccc; border-radius: 10px; background-color: white; }")
        results_layout = QVBoxLayout()
        results_panel.setLayout(results_layout)
        
        title = QLabel("Inspection Results")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 15px;")
        results_layout.addWidget(title)
        
        # TOP Process Results
        top_group = QGroupBox("TOP Process (3 Components)")
        top_layout = QVBoxLayout()
        top_group.setLayout(top_layout)
        
        self.top_status_label = QLabel("Status: Not Started")
        self.top_status_label.setStyleSheet("font-weight: bold; color: #666;")
        top_layout.addWidget(self.top_status_label)
        
        top_names = ["Antenna", "Capacitor", "Speaker"]
        self.top_component_labels = {}
        for name in top_names:
            label = QLabel(f"{name}: Pending")
            label.setStyleSheet("color: #666; padding: 3px;")
            top_layout.addWidget(label)
            self.top_component_labels[name] = label
        
        results_layout.addWidget(top_group)
        
        # BOTTOM Process Results
        bottom_group = QGroupBox("BOTTOM Process (3 Components)")
        bottom_layout = QVBoxLayout()
        bottom_group.setLayout(bottom_layout)
        
        self.bottom_status_label = QLabel("Status: Not Started")
        self.bottom_status_label.setStyleSheet("font-weight: bold; color: #666;")
        bottom_layout.addWidget(self.bottom_status_label)
        
        self.bottom_component_labels = []
        for i in range(3):
            label = QLabel(f"Component {i+1}: Pending")
            label.setStyleSheet("color: #666; padding: 3px;")
            bottom_layout.addWidget(label)
            self.bottom_component_labels.append(label)
        
        results_layout.addWidget(bottom_group)
        
        # Overall result
        self.overall_result = QLabel("Overall: Pending")
        self.overall_result.setFont(QFont("Arial", 14, QFont.Bold))
        self.overall_result.setStyleSheet("color: #2c3e50; margin: 10px;")
        self.overall_result.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.overall_result)
        # Manual Override Section
        manual_group = QGroupBox("Manual Override")
        manual_layout = QVBoxLayout()
        manual_group.setLayout(manual_layout)

        self.manual_override_checkbox = QCheckBox("Enable Manual Override")
        self.manual_override_checkbox.toggled.connect(self.toggle_manual_override)
        manual_layout.addWidget(self.manual_override_checkbox)

        # Dropdowns for manual results
        from PyQt5.QtWidgets import QComboBox
        self.manual_fields = {
            "Antenna": QComboBox(),
            "Capacitor": QComboBox(),
            "Speaker": QComboBox()
        }
        for name, combo in self.manual_fields.items():
            combo.addItems(["PASS", "FAIL"])
            combo.setEnabled(False)
            manual_layout.addWidget(QLabel(name))
            manual_layout.addWidget(combo)

        results_layout.addWidget(manual_group)
        results_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        main_layout.addWidget(results_panel)
    def toggle_manual_override(self):
        """Enable or disable manual result selection."""
        enabled = self.manual_override_checkbox.isChecked()
        for combo in self.manual_fields.values():
            combo.setEnabled(enabled)
        print(f"Manual override {'enabled' if enabled else 'disabled'}.")
    def setup_camera(self):
        """Setup camera connection"""
        try:
            #self.camera_thread.frame_ready.connect(self.update_camera_feed)
            self.camera_thread.processed_ready.connect(self.update_processed_feed)
            self.camera_status.setText("Camera: Connected")
            self.camera_status.setStyleSheet("color: #27ae60; font-size: 14px; margin: 5px;")
            
        except Exception as e:
            print(f"Camera setup error: {e}")
            self.camera_status.setText("Camera: Error")
            self.camera_status.setStyleSheet("color: #e74c3c; font-size: 14px; margin: 5px;")
            self.camera_label.setText("Camera Error")
    def load_reference_images(self):
        """Load reference images and initialize detectors"""
        try:
            # Get reference images directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            ref_dir = os.path.join(project_root, "data", "INLINE")
            
            # Load TOP references
            top_ref_path = os.path.join(ref_dir, "top_refrence.png")

            if os.path.exists(top_ref_path):
                self.top_ref = cv2.imread(top_ref_path)
                print("TOP reference image LOADED")
            
                
                # Define ROI coordinates for 3 components in TOP
                # TODO: Adjust these coordinates based on your actual setup
                self.top_rois = {
                'Antenna': (172, 112, 105, 64),
                'Capacitor': (561, 141, 194, 153),
                'Speaker': (803, 221, 165, 198),
                    }
                
            else:
                print("⚠️ TOP reference image not found")
            
            # Load BOTTOM references
            bottom_ref_path = os.path.join(ref_dir, "bottom_reference.png")

            
            if os.path.exists(bottom_ref_path):
                self.bottom_ref = cv2.imread(bottom_ref_path)
                # Define ROI coordinates for 3 components in BOTTOM
                # TODO: Adjust these coordinates based on your actual setup
                self.bottom_rois= {
            'Plate': (830, 199, 346, 419),
                }
                from algo_bottom import SobelBottom 
                self.bottom_detector=SobelBottom()
                print("✅ BOTTOM detector initialized")
            else:
                print("⚠️ BOTTOM reference image not found")
                
        except Exception as e:
            print(f"Error loading reference images: {e}")
    '''
    def setup_camera(self):
        """Setup camera connection"""
        try:
            self.camera_thread.frame_ready.connect(self.update_camera_feed)
            self.camera_status.setText("Camera: Ready")
            self.camera_status.setStyleSheet("color: #27ae60; font-size: 14px; margin: 5px;")
        except Exception as e:
            print(f"Camera setup error: {e}")
            self.camera_status.setText("Camera: Error")
            self.camera_status.setStyleSheet("color: #e74c3c; font-size: 14px; margin: 5px;")
    '''
    @pyqtSlot(np.ndarray)
    def update_camera_feed(self, frame):
        """Update camera display"""
        try:
            # Add process indicator
            if self.current_process == "TOP":
                cv2.putText(frame, "TOP PROCESS", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif self.current_process == "BOTTOM":
                cv2.putText(frame, "BOTTOM PROCESS", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
        """Update camera settings"""
        self.camera_thread.update_settings(
            flip_h=self.flip_horizontal.isChecked(),
            flip_v=self.flip_vertical.isChecked()
        )
    
    def scan_qr_code(self):
        """Scan QR code"""
        QMessageBox.information(self, "QR Scanner", "QR scanning will be implemented.")
    
    def submit_barcode(self):
        """Submit and validate barcode"""
        barcode = self.barcode_input.text().strip()
        if not barcode:
            QMessageBox.warning(self, "Error", "Please enter a barcode.")
            return
        
        if self.validate_barcode(barcode):
            self.barcode = barcode
            self.barcode_display.setText(f"Barcode: {barcode}")
            self.barcode_display.setStyleSheet("background-color: #d4edda; padding: 8px; border: 2px solid #c3e6cb; border-radius: 5px;")
            self.start_top_button.setEnabled(True)
            QMessageBox.information(self, "Success", "Barcode validated! Ready to start TOP process.")
    
    def validate_barcode(self, barcode):
        """Validate barcode using new CHIPINSPECTION API."""
        import requests
        from LOGGERINLINE import JSONLogger
        self.logger = JSONLogger(barcode_id=barcode, station=self.station, process_name="INLINE_INSPECTION_TOP")

        api_url = "http://127.0.0.1:5000/api/CHIPINSPECTION"
        try:
            response = requests.get(f"{api_url}?barcode={barcode}", timeout=5)

            if response.status_code == 404:
                print(f"[INFO] New barcode detected: {barcode}")
                self.logger.log_step("Barcode Validation", "PASS", {"reason": "New barcode"})
                return True

            elif response.status_code == 200:
                data = response.json()
                records = data.get("data", [])
                if records:
                    latest = records[0]
                    info_text = (
                        f"<b>Barcode:</b> {latest.get('Barcode', barcode)}<br>"
                        f"<b>Process ID:</b> {latest.get('Process_id', '-')}"
                    )

                    dialog = QDialog(self)
                    dialog.setWindowTitle("Duplicate Barcode Found")
                    layout = QVBoxLayout(dialog)
                    label = QLabel(info_text)
                    label.setTextFormat(1)
                    layout.addWidget(label)

                    append_btn = QPushButton("Append")
                    update_btn = QPushButton("Update")
                    cancel_btn = QPushButton("Cancel")
                    layout.addWidget(append_btn)
                    layout.addWidget(update_btn)
                    layout.addWidget(cancel_btn)

                    def append_action(): dialog.done(1)
                    def update_action(): dialog.done(2)
                    def cancel_action(): dialog.done(0)
                    append_btn.clicked.connect(append_action)
                    update_btn.clicked.connect(update_action)
                    cancel_btn.clicked.connect(cancel_action)

                    result = dialog.exec_()

                    if result == 1:
                        self.logger.set_type(1)
                        self.logger.log_step("Barcode Validation", "PASS", {"mode": "append"})
                        return True
                    elif result == 2:
                        self.logger.set_type(2)
                        self.logger.log_step("Barcode Validation", "PASS", {"mode": "update"})
                        return True
                    else:
                        self.logger.log_step("Barcode Validation", "SKIPPED", {"mode": "cancel"})
                        return False
                else:
                    return True
            else:
                QMessageBox.warning(self, "Validation Failed", f"Unexpected response: {response.status_code}")
                self.logger.log_step("Barcode Validation", "FAIL", {"code": response.status_code})
                return False

        except requests.ConnectionError:
            QMessageBox.critical(self, "Connection Error", "Could not connect to API server.")
            self.logger.log_step("Barcode Validation", "FAIL", {"reason": "Connection Error"})
            return False

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Validation error: {e}")
            self.logger.log_step("Barcode Validation", "FAIL", {"reason": str(e)})
            return False
    def stop_inspection(self):
        """Stop the current inspection and reset the state."""
        reply = QMessageBox.question(
            self, "Stop Inspection",
            "Are you sure you want to stop the current inspection?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                # Stop camera safely
                self.camera_thread.stop_camera()
                self.camera_active = False
                self.current_process = None

                # Reset display
                self.camera_label.clear()
                self.camera_label.setText("Inspection Stopped.\n\nCamera feed halted.")
                self.camera_label.setStyleSheet("""
                    background-color: #2c3e50; 
                    color: white; 
                    font-size: 18px; 
                    border-radius: 8px;
                """)

                # Reset status labels
                self.top_status_label.setText("Status: Not Started")
                self.top_status_label.setStyleSheet("font-weight: bold; color: #666;")

                self.bottom_status_label.setText("Status: Not Started")
                self.bottom_status_label.setStyleSheet("font-weight: bold; color: #666;")

                self.camera_status.setText("Camera: Stopped")
                self.camera_status.setStyleSheet("color: #e67e22; font-size: 14px; margin: 5px;")
                self.capture_top_button.setEnabled(False)
                self.submit_top_button.setEnabled(False)
                self.capture_bottom_button.setEnabled(False)
                self.submit_bottom_button.setEnabled(False)
                self.start_top_button.setEnabled(True)
                self.start_bottom_button.setEnabled(False)

                print("🛑 Inspection stopped manually by user.")

            except Exception as e:
                QMessageBox.warning(self, "Stop Error", f"Failed to stop inspection: {e}")
    def start_top_process(self):
        """Start TOP process"""
        self.current_process = "TOP"
        self.camera_active = True
        self.show_processed=False
        
        if not self.camera_thread.running:
            if not self.camera_thread.start_camera(0):
                QMessageBox.critical(self, "Camera Error", "Unable to start camera.")
                return
        
        self.camera_status.setText("Camera: Running (TOP Process)")
        self.camera_status.setStyleSheet("color: #27ae60; font-size: 14px; margin: 5px;")
        
        self.top_status_label.setText("Status: In Progress")
        self.top_status_label.setStyleSheet("font-weight: bold; color: #3498db;")
        
        self.start_top_button.setEnabled(False)
        self.capture_top_button.setEnabled(True)
        
        print("🔝 TOP process started")
    
    def capture_top(self):
        """Capture current frame and perform analysis"""
        self.show_processed = True

        # Capture the latest frame from camera
        ret, frame = self.camera_thread.camera.read()
        if not ret:
            QMessageBox.warning(self, "Camera Error", "Failed to capture frame.")
            return
        
        # Freeze frame
        self.captured_frame = frame.copy()
        # Stop continuous updates — freeze live feed
        self.camera_thread.running = False


        # Perform analysis on captured frame
        processed, concatenated = self.camera_thread.process_frame(self.captured_frame)
        try:
    # Directly use the updated top_results from CameraThread
            roi_results = self.top_results
            if not roi_results:
                print("[WARN] No valid results returned from TOP detector.")
                self.top_status_label.setText("Status: ERROR ⚠️")
                self.top_status_label.setStyleSheet("font-weight: bold; color: #f39c12;")
                return

            all_pass = True
            for name, status in roi_results.items():
                label_text = f"{name}: {status}"
                color = "#27ae60" if status.upper() == "PASS" else "#e74c3c"
                self.top_component_labels[name].setText(label_text)
                self.top_component_labels[name].setStyleSheet(
                    f"color: {color}; padding: 3px; font-weight: bold;"
                )
                if status.upper() != "PASS":
                    all_pass = False

            # Update TOP status
            if all_pass:
                self.top_status_label.setText("Status: PASS ✅")
                self.top_status_label.setStyleSheet("font-weight: bold; color: #27ae60;")
            else:
                self.top_status_label.setText("Status: FAIL ❌")
                self.top_status_label.setStyleSheet("font-weight: bold; color: #e74c3c;")

            # Enable Submit button, disable capture
            self.capture_top_button.setEnabled(False)
            self.submit_top_button.setEnabled(True)


        except Exception as e:
            print(f"[ERROR] Failed to update TOP inline results: {e}")
            self.top_status_label.setText("Status: ERROR ⚠️")
            self.top_status_label.setStyleSheet("font-weight: bold; color: #f39c12;")
        self.update_processed_feed(self.captured_frame, processed, concatenated)
        '''
        
        if not self.camera_thread.camera:
            QMessageBox.warning(self, "Camera Error", "Camera not available.")
            return

        #from preprocess import FramePreprocessor
        #processor = FramePreprocessor()  # capture burst of frames
        try:
            # Step 1: Capture averaged frame
            avg_frame = "/Users/veervardhansingh/NOVUS/AI_VDI_System/data/INLINE/TOP.png" #processor.preprocess(self.camera_thread.camera)
            self.top_frame = cv2.imread(avg_frame)
            self.top_detector = ImageInspectionPipeline()
            # Step 2: Analyze averaged frame with the detector
            if not self.top_detector:
                QMessageBox.warning(self, "Detector Error", "TOP detector not initialized.")
                return


            
            results = self.top_detector.run(ref_img=self.top_ref,input_img=self.top_frame,roi_definitions=self.top_rois)
            self.top_results = results

            # Step 3: Annotate
            vis_frame=results['annotated']
            print(results["results"])


            # Step 4: Concatenate averaged + annotated frame side-by-side
            try:
                from src.ui.advanced_inspection_window import ImageProcessor
                img_processor = ImageProcessor()
                concat_frame = img_processor.concatenate_frames(self.top_frame, vis_frame)
            except Exception as e:
                print(f"Concatenation error: {e}")
                concat_frame = vis_frame
            
            # Step 5: Update UI for component statuses
            all_present = True
            roi_results = results["results"]
            for i, (roi_name, status) in enumerate(roi_results.items()):
                status_text = f"Component {i+1}: {status}"
                color = "color: #27ae60;" if status == "PASS" else "color: #e74c3c;"
                self.top_component_labels[i].setText(status_text)
                self.top_component_labels[i].setStyleSheet(f"{color} padding: 3px; font-weight: bold;")

            try:
                # Pause live camera updates
                try:
                    self.camera_thread.running = False
                    self.camera_thread.quit()
                    self.camera_thread.wait(1000)
                except Exception:
                    pass  # already disconnected

                # Convert and display manually
                if concat_frame is None or not isinstance(concat_frame, np.ndarray):
                    raise ValueError("Concatenated frame is invalid or empty.")
                self.update_camera_feed(concat_frame)
                cv2.imwrite(f"concatenated_{self.current_process.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg", concat_frame)
                print("✅ Concatenated frame displayed successfully.")
            except Exception as e:
                print(f"⚠️ Error displaying concatenated frame: {e}")
            # Step 6: Display concatenated stream in the camera window
            rgb_image = cv2.cvtColor(concat_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)

            
            
            # Step 7: Update process status
            status = "PASS" if all_present else "FAIL"
            self.top_status_label.setText(f"Status: {status}")
            self.top_status_label.setStyleSheet(f"font-weight: bold; color: {'#27ae60' if all_present else '#e74c3c'};")

            self.capture_top_button.setEnabled(False)
            self.submit_top_button.setEnabled(True)

            QMessageBox.information(
                self, "TOP Analysis Complete",
                f"TOP process analyzed using averaged + annotated view!\n\nResult:\n\n"
                f"Ready to submit to API."
            )

        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Error during frame averaging or streaming:\n{e}")
            '''
    def complete_inspection(self):
        """Mark the current inspection as complete and reset for a new part."""
        try:
        
            self.top_results.clear()
            self.bottom_results.clear()
            self.barcode = ""

            # Reset UI
            self.camera_label.setText("Ready for next inspection.\n\nScan new barcode to begin.")
            self.camera_label.setStyleSheet("""
                background-color: #2c3e50; 
                color: white; 
                font-size: 18px; 
                border-radius: 8px;
            """)
            self.camera_thread.stop_camera()

            self.top_status_label.setText("Status: Not Started")
            self.top_status_label.setStyleSheet("font-weight: bold; color: #666;")

            self.bottom_status_label.setText("Status: Not Started")
            self.bottom_status_label.setStyleSheet("font-weight: bold; color: #666;")

            for lbl in self.top_component_labels.values():
                lbl.setText("Pending")
                lbl.setStyleSheet("color: #666; padding: 3px;")

            for lbl in self.bottom_component_labels:
                lbl.setText("Pending")
                lbl.setStyleSheet("color: #666; padding: 3px;")

            self.overall_result.setText("Overall: Pending")
            self.overall_result.setStyleSheet("color: #2c3e50; margin: 10px;")

            # Enable barcode entry again
            self.start_top_button.setEnabled(False)
            self.start_bottom_button.setEnabled(False)
            self.capture_top_button.setEnabled(False)
            self.capture_bottom_button.setEnabled(False)
            self.submit_top_button.setEnabled(False)
            self.submit_bottom_button.setEnabled(False)
            self.submit_barcode_button.setEnabled(True)
            self.barcode_input.setEnabled(True)
            self.barcode_display.setText("No barcode entered")

            print("🔁 System reset: Ready for new barcode.")

        except Exception as e:
            QMessageBox.critical(self, "Reset Error", f"Error while completing inspection: {e}")
            print(f"[ERROR] complete_inspection failed: {e}")

    
    def submit_top_to_api(self):
        """Submit TOP inspection results to the new dynamic API."""
        from LOGGERINLINE import JSONLogger

        if not self.top_results or not isinstance(self.top_results, dict):
            QMessageBox.warning(self, "No Data", "No valid TOP results to submit.")
            return

        # ---- 1️⃣ Evaluate automatic results ----
        antenna_status = self.top_results.get("Antenna", "FAIL")
        capacitor_status = self.top_results.get("Capacitor", "FAIL")
        speaker_status = self.top_results.get("Speaker", "FAIL")

        auto_pass = all(s == "PASS" for s in [antenna_status, capacitor_status, speaker_status])

        # ---- 2️⃣ Handle manual overrides ----
        manual_enabled = self.manual_override_checkbox.isChecked()
        manual_antenna = self.manual_fields["Antenna"].currentText() if manual_enabled else None
        manual_capacitor = self.manual_fields["Capacitor"].currentText() if manual_enabled else None
        manual_speaker = self.manual_fields["Speaker"].currentText() if manual_enabled else None

        if manual_enabled:
            print("[INFO] Manual override is active.")
            manual_pass = all(s == "PASS" for s in [manual_antenna, manual_capacitor, manual_speaker])
        else:
            manual_pass = auto_pass

        overall_result = "PASS" if manual_pass else "FAIL"

        # ---- 3️⃣ Construct server payload ----
        payload = {
            "Barcode": self.barcode,
            "DT": datetime.now().isoformat(),
            "Process_id": 2,  # INLINE process
            "Station_ID": self.station,
            "Antenna": 1 if antenna_status == "PASS" else 0,
            "Capacitor": 1 if capacitor_status == "PASS" else 0,
            "Speaker": 1 if speaker_status == "PASS" else 0,
            "Result": 1 if auto_pass else 0,
            "ManualAntenna": 1 if manual_antenna == "PASS" else 0,
            "ManualCapacitor": 1 if manual_capacitor == "PASS" else 0,
            "ManualSpeaker": 1 if manual_speaker == "PASS" else 0,
            "ManualResult": 1 if manual_pass else 0
        }

        # ---- 4️⃣ Initialize and log ----
        if not self.logger:
            self.logger = JSONLogger(self.barcode, self.station, "INLINE_INSPECTION_BOTTOM")

        self.logger.log_step("TOP Inspection Analysis", overall_result)
        self.logger.log_payload(payload)

        # ---- 5️⃣ Send to server ----
        try:
            api_url = "http://127.0.0.1:5000/api/INLINEINSPECTIONBOTTOM"
            response = requests.post(api_url, json=payload, timeout=5)

            if response.status_code in (200, 201):
                QMessageBox.information(
                    self,
                    "TOP Submitted",
                    f"✅ TOP process submitted successfully!\nStatus: {overall_result}"
                )
                self.top_completed = True
                self.submit_top_button.setEnabled(False)
                self.start_bottom_button.setEnabled(True)
                self.logger.log_step("Submit TOP", "PASS", {"response": response.status_code})
                self.logger.set_final_status(overall_result)
                print(f"✅ TOP submitted to server → {overall_result}")

            else:
                QMessageBox.warning(
                    self,
                    "API Error",
                    f"⚠️ Server returned {response.status_code}: {response.text}"
                )
                self.logger.log_step("Submit TOP", "FAIL", {"response": response.status_code})

        except requests.ConnectionError:
            QMessageBox.critical(self, "Connection Error", "Could not reach API server.")
            self.logger.log_step("Submit TOP", "FAIL", {"reason": "Connection Error"})

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {e}")
            self.logger.log_step("Submit TOP", "FAIL", {"exception": str(e)})
    
    def start_bottom_process(self):
        """Start  process"""
        self.current_process = "BOTTOM"
        self.camera_active = True
        self.show_processed=False
        
        if not self.camera_thread.running:
            if not self.camera_thread.start_camera(0):
                QMessageBox.critical(self, "Camera Error", "Unable to start camera.")
                return
        
        self.camera_status.setText("Camera: Running (BOTTOM Process)")
        self.camera_status.setStyleSheet("color: #27ae60; font-size: 14px; margin: 5px;")
        
        self.bottom_status_label.setText("Status: In Progress")
        self.bottom_status_label.setStyleSheet("font-weight: bold; color: #3498db;")
        
        self.start_bottom_button.setEnabled(False)
        self.capture_bottom_button.setEnabled(True)
        
        print("⬇️ BOTTOM process started")
    
    def capture_bottom(self):
        """Capture current frame and perform analysis"""
        # Stop continuous updates — freeze live feed
    

        self.show_processed = True

        # Capture the latest frame from camera
        ret, frame = self.camera_thread.camera.read()
        if not ret:
            QMessageBox.warning(self, "Camera Error", "Failed to capture frame.")
            return
        
        # Freeze frame
        self.captured_frame = frame.copy()
        self.camera_thread.running = False


        # Perform analysis on captured frame
        processed, concatenated = self.camera_thread.process_frame(self.captured_frame)
        self.update_processed_feed(self.captured_frame, processed, concatenated)
        try:
            roi_results = self.bottom_results  # populated by SobelBottom
            if not roi_results or not isinstance(roi_results, dict):
                print("[WARN] No valid results returned from BOTTOM detector.")
                self.bottom_status_label.setText("Status: ERROR ⚠️")
                self.bottom_status_label.setStyleSheet("font-weight: bold; color: #f39c12;")
                return

            all_pass = True

            # Update UI labels for Screw and Plate
            for name, status in roi_results.items():
                label_text = f"{name}: {status}"
                color = "#27ae60" if status.upper() == "PASS" else "#e74c3c"

                # Handle dynamic lookup (Screw, Plate)
                if name in self.bottom_component_labels:
                    self.bottom_component_labels[name].setText(label_text)
                    self.bottom_component_labels[name].setStyleSheet(
                        f"color: {color}; padding: 3px; font-weight: bold;"
                    )
                else:
                    print(f"[WARN] Unknown bottom label: {name}")

                if status.upper() != "PASS":
                    all_pass = False

            # Update overall bottom status
            if all_pass:
                self.bottom_status_label.setText("Status: PASS ✅")
                self.bottom_status_label.setStyleSheet("font-weight: bold; color: #27ae60;")
            else:
                self.bottom_status_label.setText("Status: FAIL ❌")
                self.bottom_status_label.setStyleSheet("font-weight: bold; color: #e74c3c;")

            # Enable Submit button after analysis
            self.capture_bottom_button.setEnabled(False)
            self.submit_bottom_button.setEnabled(True)

            print(f"✅ Inline BOTTOM analysis complete → {roi_results}")

        except Exception as e:
            print(f"[ERROR] Failed to update BOTTOM results: {e}")
            self.bottom_status_label.setText("Status: ERROR ⚠️")
            self.bottom_status_label.setStyleSheet("font-weight: bold; color: #f39c12;")
        """Capture and analyze BOTTOM frame"""
      
    '''   
        # Update UI
        all_present = True
        for i, result in enumerate(results):
            status_text = f"Component {i+1}: {result['status']}"
            color = "color: #27ae60;" if result['pass'] else "color: #e74c3c;"
            self.bottom_component_labels[i].setText(status_text)
            self.bottom_component_labels[i].setStyleSheet(f"{color} padding: 3px; font-weight: bold;")
            
            if not result['detected']:
                all_present = False
        
        # Show visualization
        vis_frame = self.bottom_detector.visualize_results(self.bottom_frame, results)
        rgb_image = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled_pixmap)
        
        self.capture_bottom_button.setEnabled(False)
        self.submit_bottom_button.setEnabled(True)
        
        status = "PASS" if all_present else "FAIL"
        self.bottom_status_label.setText(f"Status: {status}")
        self.bottom_status_label.setStyleSheet(f"font-weight: bold; color: {'#27ae60' if all_present else '#e74c3c'};")
        
        QMessageBox.information(self, "BOTTOM Analysis Complete", 
                                f"BOTTOM process analyzed!\n\nResult: {status}\n\n"
                                f"Ready to submit to API.")
        else:
            QMessageBox.warning(self, "Detector Error", "BOTTOM detector not initialized.")
        '''
    def submit_bottom_to_api(self):
        """Submit BOTTOM inspection results to the new dynamic API."""
        from logger import JSONLogger

        if not self.bottom_results or not isinstance(self.bottom_results, dict):
            QMessageBox.warning(self, "No Data", "No valid BOTTOM results to submit.")
            return

        # ---- 1️⃣ Extract result statuses ----
        plate_status = self.bottom_results.get("Plate", "FAIL")
        screw_status = self.bottom_results.get("Screw", "FAIL")

        auto_pass = all(s == "PASS" for s in [plate_status, screw_status])
        overall_result = "PASS" if auto_pass else "FAIL"

        # ---- 2️⃣ Handle manual override ----
        manual_enabled = self.manual_override_checkbox.isChecked()

        if manual_enabled:
            print("[INFO] Manual override enabled for BOTTOM inspection.")
            # You can reuse top dropdowns or add separate ones later
            manual_plate = self.manual_fields["Antenna"].currentText()  # TEMP: reuse existing combo
            manual_screw = self.manual_fields["Capacitor"].currentText()  # TEMP: reuse existing combo
            manual_pass = all(s == "PASS" for s in [manual_plate, manual_screw])
        else:
            manual_plate = None
            manual_screw = None
            manual_pass = auto_pass

        manual_result = "PASS" if manual_pass else "FAIL"

        # ---- 3️⃣ Construct API payload ----
        payload = {
            "Barcode": self.barcode,
            "DT": datetime.now().isoformat(),
            "Process_id": 2,  # INLINE process
            "Station_ID": self.station,
            "Plate": 1 if plate_status == "PASS" else 0,
            "Screw": 1 if screw_status == "PASS" else 0,
            "Result": 1 if auto_pass else 0,
            "ManualPlate": 1 if manual_plate == "PASS" else 0,
            "ManualScrew": 1 if manual_screw == "PASS" else 0,
            "ManualResult": 1 if manual_result == "PASS" else 0
        }

        # ---- 4️⃣ Initialize logger if missing ----
        if not self.logger:
            self.logger = JSONLogger(self.barcode, self.station, "INLINE_INSPECTION_TOP")

        # Update logger for bottom process
        self.logger.process_name = "INLINE_INSPECTION_TOP"
        self.logger.log_step("BOTTOM Inspection Analysis", overall_result)
        self.logger.log_payload(payload)

        # ---- 5️⃣ Send payload to server ----
        try:
            api_url = "http://127.0.0.1:5000/api/INLINEINSPECTIONTOP"
            response = requests.post(api_url, json=payload, timeout=5)

            if response.status_code in (200, 201):
                QMessageBox.information(
                    self,
                    "BOTTOM Submitted",
                    f"✅ BOTTOM process submitted successfully!\nStatus: {overall_result}"
                )
                self.submit_bottom_button.setEnabled(False)
                self.bottom_completed = True

                # Mark completion in logs
                self.logger.log_step("Submit BOTTOM", "PASS", {"response": response.status_code})
                self.logger.set_final_status(overall_result)
                print(f"✅ BOTTOM submitted → {overall_result}")

                # Update overall inspection result
                self.calculate_overall_result()
                self.complete_inspection()
                self.back_to_main()

            else:
                QMessageBox.warning(
                    self,
                    "API Error",
                    f"⚠️ Server returned {response.status_code}: {response.text}"
                )
                self.logger.log_step("Submit BOTTOM", "FAIL", {"response": response.status_code})

        except requests.ConnectionError:
            QMessageBox.critical(self, "Connection Error", "Could not reach API server.")
            self.logger.log_step("Submit BOTTOM", "FAIL", {"reason": "Connection Error"})

        except Exception as e:
            self.back_to_main()
            self.logger.log_step("Submit BOTTOM", "FAIL", {"exception": str(e)})
       

       
    
    def calculate_overall_result(self):
        """Calculate and display overall inspection result"""
        if not self.top_completed or not self.bottom_completed:
            return
        
        # Check if all components passed
        top_pass = all(r['detected'] for r in self.top_results)
        bottom_pass = all(r['detected'] for r in self.bottom_results)
        
        overall = "PASS" if (top_pass and bottom_pass) else "FAIL"
        
        self.overall_result.setText(f"Overall: {overall}")
        if overall == "PASS":
            self.overall_result.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 16px; margin: 10px;")
        else:
            self.overall_result.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 16px; margin: 10px;")
        
        print(f"📊 Overall Result: {overall}")
    
    def back_to_main(self):
        """Return to main menu"""
        reply = QMessageBox.question(self, "Back to Main", 
                                   "Return to main menu?\n\nAny unsaved data will be lost.",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.camera_thread.stop_camera()
            self.window_closed.emit()
            self.close()
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.camera_thread.stop_camera()
        self.window_closed.emit()
        event.accept()


def main():
    """Main function for testing"""
    app = QApplication(sys.argv)
    window = DualProcessInspectionWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()