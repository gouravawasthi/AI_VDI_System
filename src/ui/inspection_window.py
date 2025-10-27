"""
Inspection Window - Main inspection interface for AI VDI System
"""

import sys
import os
import csv
import requests
from datetime import datetime
from typing import Optional, Dict, Any
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, 
                            QPushButton, QLabel, QFrame, QSpacerItem, QSizePolicy,
                            QGroupBox, QSlider, QCheckBox, QLineEdit, QTextEdit,
                            QProgressBar, QMessageBox, QComboBox, QScrollArea)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import QPixmap, QFont, QImage
import cv2
import numpy as np


class CameraThread(QThread):
    """Thread for handling camera operations"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.camera = None
        self.running = False
        self.flip_horizontal = False
        self.flip_vertical = False
        self.exposure = 0
        self.white_balance = 5000
        
    def start_camera(self, camera_id=0):
        """Start camera capture"""
        try:
            self.camera = cv2.VideoCapture(camera_id)
            if self.camera and self.camera.isOpened():
                self.running = True
                self.start()
                return True
            else:
                if self.camera:
                    self.camera.release()
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
            self.wait(3000)  # Wait up to 3 seconds
    
    def update_settings(self, flip_h=False, flip_v=False, exposure=0, wb=5000):
        """Update camera settings"""
        self.flip_horizontal = flip_h
        self.flip_vertical = flip_v
        self.exposure = exposure
        self.white_balance = wb
        
        if self.camera:
            # Apply camera settings
            self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure)
            # Note: White balance might not be supported on all cameras
    
    def run(self):
        """Main camera loop"""
        while self.running and self.camera and self.camera.isOpened():
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Apply transformations
                    if self.flip_horizontal and self.flip_vertical:
                        frame = cv2.flip(frame, -1)
                    elif self.flip_horizontal:
                        frame = cv2.flip(frame, 1)
                    elif self.flip_vertical:
                        frame = cv2.flip(frame, 0)
                    
                    self.frame_ready.emit(frame)
                else:
                    break
            except Exception as e:
                print(f"Camera read error: {e}")
                break
            self.msleep(33)  # ~30 FPS


class InspectionWindow(QWidget):
    """Main inspection window for product quality inspection"""
    
    inspection_complete = pyqtSignal(dict)
    window_closed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__()
        self.parent_window = parent
        self.barcode = ""
        self.current_side = 0
        self.inspection_sides = ["Front", "Back", "Left", "Right", "Top", "Bottom"]
        self.inspection_results = {}
        self.inspection_start_time = None
        self.side_start_time = None
        self.camera_thread = CameraThread()
        self.init_ui()
        self.setup_camera()
        
    def init_ui(self):
        """Initialize the inspection interface"""
        self.setWindowTitle("AI VDI System - Inspection")
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
                padding: 10px 20px;
                font-size: 14px;
                border-radius: 5px;
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
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLineEdit {
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
            }
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        
        # Main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left panel - Controls
        self.create_control_panel(main_layout)
        
        # Center panel - Camera feed
        self.create_camera_panel(main_layout)
        
        # Right panel - Inspection progress and results
        self.create_inspection_panel(main_layout)
    
    def create_control_panel(self, main_layout):
        """Create the control panel on the left"""
        control_panel = QFrame()
        control_panel.setMaximumWidth(350)
        control_panel.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 5px; }")
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        
        # Title
        title = QLabel("Control Panel")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
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
        
        self.scan_qr_button = QPushButton("Scan QR Code")
        self.scan_qr_button.clicked.connect(self.scan_qr_code)
        button_layout.addWidget(self.scan_qr_button)
        
        self.submit_barcode_button = QPushButton("Submit")
        self.submit_barcode_button.clicked.connect(self.submit_barcode)
        button_layout.addWidget(self.submit_barcode_button)
        
        barcode_layout.addLayout(button_layout)
        
        # Barcode display
        self.barcode_display = QLabel("No barcode entered")
        self.barcode_display.setStyleSheet("background-color: white; padding: 5px; border: 1px solid #ddd;")
        barcode_layout.addWidget(self.barcode_display)
        
        layout.addWidget(barcode_group)
    
    def create_camera_settings(self, layout):
        """Create camera settings section"""
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QVBoxLayout()
        camera_group.setLayout(camera_layout)
        
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
        """Create camera display panel"""
        camera_panel = QFrame()
        camera_panel.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 5px; }")
        camera_layout = QVBoxLayout()
        camera_panel.setLayout(camera_layout)
        
        # Camera feed
        self.camera_label = QLabel("Camera Feed")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: black; color: white; font-size: 18px;")
        camera_layout.addWidget(self.camera_label)
        
        # Camera status
        self.camera_status = QLabel("Camera: Disconnected")
        self.camera_status.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.camera_status)
        
        main_layout.addWidget(camera_panel)
    
    def create_inspection_panel(self, main_layout):
        """Create inspection progress and results panel"""
        inspection_panel = QFrame()
        inspection_panel.setMaximumWidth(350)
        inspection_panel.setStyleSheet("QFrame { border: 1px solid #ccc; border-radius: 5px; }")
        inspection_layout = QVBoxLayout()
        inspection_panel.setLayout(inspection_layout)
        
        # Title
        title = QLabel("Inspection Progress")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
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
        self.current_side_label.setFont(QFont("Arial", 12, QFont.Bold))
        progress_layout.addWidget(self.current_side_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, len(self.inspection_sides))
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        # Side status
        self.side_status_layout = QVBoxLayout()
        for side in self.inspection_sides:
            side_label = QLabel(f"{side}: Pending")
            side_label.setStyleSheet("color: #666; padding: 2px;")
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
        results_layout.addWidget(self.overall_result)
        
        # Detailed results
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        self.results_layout = QVBoxLayout()
        scroll_widget.setLayout(self.results_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)
        results_layout.addWidget(scroll_area)
        
        # Time information
        self.time_info = QLabel("Time: 00:00")
        results_layout.addWidget(self.time_info)
        
        layout.addWidget(results_group)
    
    def setup_camera(self):
        """Setup camera connection"""
        try:
            if self.camera_thread.start_camera(0):
                self.camera_thread.frame_ready.connect(self.update_camera_feed)
                self.camera_status.setText("Camera: Connected")
                self.camera_status.setStyleSheet("color: green;")
            else:
                self.camera_status.setText("Camera: Failed to connect")
                self.camera_status.setStyleSheet("color: red;")
                # Show placeholder image
                self.camera_label.setText("Camera Not Available")
        except Exception as e:
            print(f"Camera setup error: {e}")
            self.camera_status.setText("Camera: Error")
            self.camera_status.setStyleSheet("color: red;")
            self.camera_label.setText("Camera Error")
    
    @pyqtSlot(np.ndarray)
    def update_camera_feed(self, frame):
        """Update camera feed display"""
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit label
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Camera feed update error: {e}")
    
    def update_camera_settings(self):
        """Update camera settings based on UI controls"""
        self.camera_thread.update_settings(
            flip_h=self.flip_horizontal.isChecked(),
            flip_v=self.flip_vertical.isChecked(),
            exposure=self.exposure_slider.value(),
            wb=self.wb_slider.value()
        )
    
    def scan_qr_code(self):
        """Scan QR code from camera feed"""
        # TODO: Implement QR code scanning
        QMessageBox.information(self, "QR Scanner", "QR Code scanning functionality will be implemented here.")
    
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
            self.start_inspection_button.setEnabled(True)
            QMessageBox.information(self, "Success", "Barcode validated successfully!")
        else:
            QMessageBox.critical(self, "Validation Failed", 
                               "Barcode was rejected in previous process. Inspection cannot proceed.")
    
    def validate_barcode(self, barcode):
        """Validate barcode with server API"""
        try:
            # TODO: Replace with actual API endpoint
            # response = requests.get(f"http://your-api-server.com/validate/{barcode}")
            # return response.status_code == 200 and response.json().get('valid', False)
            
            # Mock validation - always returns True for demo
            return True
        except Exception as e:
            print(f"API validation error: {e}")
            return False
    
    def start_inspection(self):
        """Start the inspection process"""
        self.inspection_start_time = datetime.now()
        self.current_side = 0
        self.inspection_results = {}
        
        # Enable controls
        self.start_inspection_button.setEnabled(False)
        self.next_side_button.setEnabled(True)
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
            
            # Update side status
            for i, label in enumerate(self.side_status_layout.children()):
                if hasattr(label, 'setText'):
                    if i == self.current_side:
                        label.setText(f"{self.inspection_sides[i]}: In Progress")
                        label.setStyleSheet("color: blue; font-weight: bold; padding: 2px;")
    
    def next_side(self):
        """Move to next side of inspection"""
        if self.current_side < len(self.inspection_sides):
            side_name = self.inspection_sides[self.current_side]
            
            # Simulate inspection result
            result = self.perform_side_inspection(side_name)
            
            # Record result
            side_time = (datetime.now() - self.side_start_time).total_seconds()
            self.inspection_results[side_name] = {
                'result': result,
                'time': side_time,
                'timestamp': datetime.now()
            }
            
            # Update UI
            self.update_side_status(self.current_side, result)
            self.current_side += 1
            self.progress_bar.setValue(self.current_side)
            
            if self.current_side < len(self.inspection_sides):
                self.start_side_inspection()
            else:
                self.complete_inspection()
    
    def perform_side_inspection(self, side_name):
        """Perform actual inspection logic for a side"""
        # TODO: Implement actual ML inference
        # This is a mock implementation
        import random
        return "PASS" if random.random() > 0.3 else "FAIL"
    
    def update_side_status(self, side_index, result):
        """Update the status of a specific side"""
        if side_index < self.side_status_layout.count():
            label = self.side_status_layout.itemAt(side_index).widget()
            if label:
                side_name = self.inspection_sides[side_index]
                label.setText(f"{side_name}: {result}")
                if result == "PASS":
                    label.setStyleSheet("color: green; font-weight: bold; padding: 2px;")
                else:
                    label.setStyleSheet("color: red; font-weight: bold; padding: 2px;")
    
    def complete_inspection(self):
        """Complete the inspection process"""
        total_time = (datetime.now() - self.inspection_start_time).total_seconds()
        
        # Determine overall result
        failed_sides = [side for side, data in self.inspection_results.items() if data['result'] == 'FAIL']
        overall_result = "FAIL" if failed_sides else "PASS"
        
        # Update UI
        self.overall_result.setText(f"Overall: {overall_result}")
        if overall_result == "PASS":
            self.overall_result.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.overall_result.setStyleSheet("color: red; font-weight: bold;")
        
        self.time_info.setText(f"Total Time: {total_time:.1f}s")
        
        # Display detailed results
        self.display_detailed_results(failed_sides)
        
        # Log results
        self.log_inspection_results(overall_result, total_time, failed_sides)
        
        # Disable controls
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
                result_label.setStyleSheet("color: green; padding: 2px;")
            else:
                result_label.setStyleSheet("color: red; padding: 2px;")
            self.results_layout.addWidget(result_label)
    
    def log_inspection_results(self, overall_result, total_time, failed_sides):
        """Log inspection results to CSV file"""
        try:
            # Ensure logs directory exists
            logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            
            # CSV file path
            csv_file = os.path.join(logs_dir, "inspection_log.csv")
            
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
                
        except Exception as e:
            print(f"Error logging results: {e}")
    
    def manual_override(self):
        """Handle manual override"""
        reply = QMessageBox.question(self, "Manual Override", 
                                   "Do you want to manually override the current inspection result?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # TODO: Implement manual override logic
            QMessageBox.information(self, "Override", "Manual override applied.")
    
    def stop_inspection(self):
        """Stop the current inspection"""
        reply = QMessageBox.question(self, "Stop Inspection", 
                                   "Are you sure you want to stop the inspection?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
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
        for i, label in enumerate(self.side_status_layout.children()):
            if hasattr(label, 'setText'):
                side_name = self.inspection_sides[i]
                label.setText(f"{side_name}: Pending")
                label.setStyleSheet("color: #666; padding: 2px;")
        
        # Reset buttons
        self.start_inspection_button.setEnabled(bool(self.barcode))
        self.next_side_button.setEnabled(False)
        self.manual_override_button.setEnabled(False)
        self.stop_inspection_button.setEnabled(False)
    
    def back_to_main(self):
        """Return to main window"""
        reply = QMessageBox.question(self, "Back to Main", 
                                   "Are you sure you want to return to the main menu? Any ongoing inspection will be lost.",
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
    """Main function for testing inspection window"""
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = InspectionWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()