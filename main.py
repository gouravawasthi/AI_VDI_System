"""
AI VDI System - Main Entry Point
Visual Defect Inspection System using AI/ML for automated quality control
"""

import sys
import os
import configparser
import logging
from pathlib import Path

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from src.ui.main_window import MainWindow
from src.core.inspection_manager import InspectionManager
from PyQt5.QtWidgets import QApplication


def setup_logging(log_level: str = 'INFO'):
    """
    Setup logging configuration for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'ai_vdi_system.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str = 'config.ini') -> dict:
    """
    Load configuration from INI file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config = configparser.ConfigParser()
    
    # Set default configuration
    default_config = {
        'camera': {
            'camera_id': '0',
            'resolution_width': '1920',
            'resolution_height': '1080',
            'fps': '30'
        },
        'ml': {
            'model_path': 'data/models/defect_model.pth',
            'model_type': 'cnn',
            'device': 'cpu',
            'threshold': '0.5',
            'target_size_width': '224',
            'target_size_height': '224'
        },
        'barcode': {
            'enabled': 'true',
            'scan_timeout': '5.0',
            'supported_formats': 'CODE128,QR'
        },
        'system': {
            'log_level': 'INFO',
            'auto_start': 'false',
            'save_inspection_images': 'true',
            'inspection_log_path': 'data/inspection_logs'
        }
    }
    
    # Load default configuration
    config.read_dict(default_config)
    
    # Try to load from file
    if os.path.exists(config_path):
        try:
            config.read(config_path)
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration")
    else:
        print(f"Config file {config_path} not found. Using default configuration")
    
    # Convert to nested dictionary
    config_dict = {}
    for section_name in config.sections():
        section = config[section_name]
        config_dict[section_name] = {}
        
        for key, value in section.items():
            # Try to convert to appropriate types
            if value.lower() in ('true', 'false'):
                config_dict[section_name][key] = value.lower() == 'true'
            elif value.replace('.', '').isdigit():
                if '.' in value:
                    config_dict[section_name][key] = float(value)
                else:
                    config_dict[section_name][key] = int(value)
            else:
                config_dict[section_name][key] = value
    
    return config_dict


def create_directory_structure():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/raw',
        'data/processed', 
        'data/models',
        'data/inspection_logs',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def main():
    """Main function to start the AI VDI System"""
    try:
        # Create necessary directories
        create_directory_structure()
        
        # Load configuration
        config = load_config()
        
        # Setup logging
        log_level = config.get('system', {}).get('log_level', 'INFO')
        setup_logging(log_level)
        
        logger = logging.getLogger('Main')
        logger.info("Starting AI VDI System")
        
        # Create Qt Application
        app = QApplication(sys.argv)
        
        # Create main window
        main_window = MainWindow()
        
        # Initialize inspection manager
        inspection_manager = InspectionManager(config)
        
        # Connect inspection manager to UI (if needed)
        # This can be expanded to connect signals/slots between UI and core logic
        
        # Show main window
        main_window.show()
        
        logger.info("AI VDI System started successfully")
        
        # Auto-start inspection if configured
        if config.get('system', {}).get('auto_start', False):
            logger.info("Auto-starting inspection process")
            inspection_manager.start_inspection()
        
        # Run application
        exit_code = app.exec_()
        
        # Cleanup
        logger.info("Shutting down AI VDI System")
        if inspection_manager.is_running:
            inspection_manager.stop_inspection()
        
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.exception("Fatal error occurred")
        sys.exit(1)


if __name__ == "__main__":
    main()