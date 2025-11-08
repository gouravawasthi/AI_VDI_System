"""
Main application window for AI VDI System - Updated with Dual Process Inspection
"""

import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QFrame, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont


class MainWindow(QMainWindow):
    """Main application window for the AI VDI System"""
    
    # Define signals
    start_inspection = pyqtSignal()
    start_inline_inspection = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.inspection_window = None
        self.inline_inspection_window = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("AI VDI System")
        self.showFullScreen()  # Set to full screen mode
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton#inlineButton {
                background-color: #2196F3;
            }
            QPushButton#inlineButton:hover {
                background-color: #0b7dda;
            }
            QPushButton#quitButton {
                background-color: #f44336;
            }
            QPushButton#quitButton:hover {
                background-color: #da190b;
            }
            QLabel {
                color: #333;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Add title
        title_label = QLabel("AI VDI System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 36, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin: 30px;")
        main_layout.addWidget(title_label)
        
        # Create brand images section
        self.create_brand_section(main_layout)
        
        # Add spacer
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Create control buttons section
        self.create_buttons_section(main_layout)
        
        # Add spacer
        main_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
    
    def create_brand_section(self, main_layout):
        """Create the brand images and text section"""
        # Create horizontal layout for brand section
        brand_layout = QHBoxLayout()
        
        # Add spacer to center the content
        brand_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # Taisys section
        taisys_frame = QFrame()
        taisys_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #3498db;
                border-radius: 10px;
                background-color: white;
                padding: 10px;
            }
        """)
        taisys_layout = QVBoxLayout()
        taisys_frame.setLayout(taisys_layout)
        
        # Taisys image
        taisys_image = QLabel()
        taisys_pixmap = self.load_brand_image("Taisys.jpeg")
        if taisys_pixmap:
            taisys_image.setPixmap(taisys_pixmap)
            taisys_image.setAlignment(Qt.AlignCenter)
        else:
            taisys_image.setText("Taisys Logo")
            taisys_image.setAlignment(Qt.AlignCenter)
            taisys_image.setStyleSheet("border: 1px dashed #ccc; min-height: 450px; min-width: 600px;")
        
        # Taisys text
        taisys_text = QLabel("Build for Taisys")
        taisys_text.setAlignment(Qt.AlignCenter)
        taisys_text.setFont(QFont("Arial", 20, QFont.Bold))
        taisys_text.setStyleSheet("color: #3498db; margin: 15px;")
        
        taisys_layout.addWidget(taisys_image)
        taisys_layout.addWidget(taisys_text)
        
        brand_layout.addWidget(taisys_frame)
        
        # Add spacer between images
        brand_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))
        
        # Avenya section
        avenya_frame = QFrame()
        avenya_frame.setStyleSheet("""
            QFrame {
                border: 2px solid #e74c3c;
                border-radius: 10px;
                background-color: white;
                padding: 10px;
            }
        """)
        avenya_layout = QVBoxLayout()
        avenya_frame.setLayout(avenya_layout)
        
        # Avenya image
        avenya_image = QLabel()
        avenya_pixmap = self.load_brand_image("Avenya.jpg")
        if avenya_pixmap:
            avenya_image.setPixmap(avenya_pixmap)
            avenya_image.setAlignment(Qt.AlignCenter)
        else:
            avenya_image.setText("Avenya Logo")
            avenya_image.setAlignment(Qt.AlignCenter)
            avenya_image.setStyleSheet("border: 1px dashed #ccc; min-height: 450px; min-width: 600px;")
        
        # Avenya text
        avenya_text = QLabel("Build by Avenya")
        avenya_text.setAlignment(Qt.AlignCenter)
        avenya_text.setFont(QFont("Arial", 20, QFont.Bold))
        avenya_text.setStyleSheet("color: #e74c3c; margin: 15px;")
        
        avenya_layout.addWidget(avenya_image)
        avenya_layout.addWidget(avenya_text)
        
        brand_layout.addWidget(avenya_frame)
        
        # Add spacer to center the content
        brand_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        main_layout.addLayout(brand_layout)
    
    def create_buttons_section(self, main_layout):
        """Create the control buttons section"""
        # Create horizontal layout for buttons
        buttons_layout = QHBoxLayout()
        
        # Add spacer to center buttons
        buttons_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        # Start Inspection button (original 6-sided inspection)
        self.start_button = QPushButton("Start Inspection\n(6-Sided)")
        self.start_button.setMinimumSize(250, 80)
        self.start_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.start_button.clicked.connect(self.on_start_clicked)
        buttons_layout.addWidget(self.start_button)
        
        # Add spacer between buttons
        buttons_layout.addItem(QSpacerItem(30, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))
        
        # Start Inline Inspection button (NEW - dual process)
        self.start_inline_button = QPushButton("Start Inspection Inline\n(Top/Bottom Components)")
        self.start_inline_button.setObjectName("inlineButton")
        self.start_inline_button.setMinimumSize(250, 80)
        self.start_inline_button.setFont(QFont("Arial", 16, QFont.Bold))
        self.start_inline_button.clicked.connect(self.on_start_inline_clicked)
        buttons_layout.addWidget(self.start_inline_button)
        
        # Add spacer between buttons
        buttons_layout.addItem(QSpacerItem(30, 20, QSizePolicy.Fixed, QSizePolicy.Minimum))
        
        # Quit button
        self.quit_button = QPushButton("Quit")
        self.quit_button.setObjectName("quitButton")
        self.quit_button.setMinimumSize(250, 80)
        self.quit_button.setFont(QFont("Arial", 18, QFont.Bold))
        self.quit_button.clicked.connect(self.on_quit_clicked)
        buttons_layout.addWidget(self.quit_button)
        
        # Add spacer to center buttons
        buttons_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        
        main_layout.addLayout(buttons_layout)
    
    def load_brand_image(self, image_name):
        """Load brand image and return scaled pixmap"""
        try:
            # Get the path to Brand_Images folder
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            image_path = os.path.join(project_root, "Brand_Images", image_name)
            
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                # Scale the image to a much larger size while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(600, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                return scaled_pixmap
            else:
                print(f"Image not found: {image_path}")
                return None
        except Exception as e:
            print(f"Error loading image {image_name}: {e}")
            return None
    
    def on_start_clicked(self):
        """Handle start button click - Original 6-sided inspection"""
        self.start_inspection.emit()
        print("Start Inspection (6-Sided) clicked")
        
        # Import and launch advanced inspection window
        from .advanced_inspection_window import AdvancedInspectionWindow
        self.inspection_window = AdvancedInspectionWindow(self)
        self.inspection_window.window_closed.connect(self.show_main_window)
        self.inspection_window.show()
        
        # Hide main window
        self.hide()
    
    def on_start_inline_clicked(self):
        """Handle start inline button click - NEW Dual Process Inspection"""
        self.start_inline_inspection.emit()
        print("Start Inspection Inline (Top/Bottom Components) clicked")
        
        # Import and launch dual process inspection window
        from .dual_process_inspection_window import DualProcessInspectionWindow
        self.inline_inspection_window = DualProcessInspectionWindow(self)
        self.inline_inspection_window.window_closed.connect(self.show_main_window_inline)
        self.inline_inspection_window.show()
        
        # Hide main window
        self.hide()
    
    def show_main_window(self):
        """Show main window when 6-sided inspection window is closed"""
        self.show()
        self.inspection_window = None
    
    def show_main_window_inline(self):
        """Show main window when inline inspection window is closed"""
        self.show()
        self.inline_inspection_window = None
    
    def on_quit_clicked(self):
        """Handle quit button click"""
        self.close()
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Cleanup resources
        if self.inspection_window:
            self.inspection_window.close()
        if self.inline_inspection_window:
            self.inline_inspection_window.close()
        event.accept()


def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()