# AI VDI System

A comprehensive AI-powered Visual Defect Inspection (VDI) system built with PyQt5 and OpenCV for real-time quality control and inspection.

## 🚀 Overview

The AI VDI System provides an industrial-grade solution for automated visual inspection with:
- **Real-time Camera Integration**: Live camera feed processing and analysis
- **Reference Image Comparison**: Advanced gradient-based defect detection
- **Interactive Mask Editor**: Professional mask creation and editing tools
- **Multi-side Inspection**: Support for 6-sided product inspection (Front, Back, Left, Right, Top, Bottom)
- **Comprehensive Logging**: Detailed audit trails and inspection reports

## 🏗️ System Architecture

```
AI_VDI_System/
├── main.py                          # Main application entry point
├── reference_ui.py                  # Reference & mask generator GUI
├── config.ini                       # System configuration
├── requirements.txt                 # Python dependencies
├── Brand_Images/                    # Company branding assets
│   ├── Taisys.jpeg
│   └── Avenya.jpg
├── src/                            # Core application modules
│   ├── __init__.py
│   ├── core/                       # Core inspection logic
│   │   ├── __init__.py
│   │   ├── barcode_handler.py      # Barcode processing
│   │   ├── camera_handler.py       # Camera operations
│   │   └── inspection_manager.py   # Main inspection controller
│   ├── ml/                         # Machine learning components
│   │   ├── __init__.py
│   │   ├── inference.py            # ML inference engine
│   │   ├── model.py                # Model definitions
│   │   └── preprocess.py           # Image preprocessing
│   └── ui/                         # User interface components
│       ├── __init__.py
│       ├── main_window.py          # Main application window
│       ├── widgets.py              # Custom UI widgets
│       ├── inspection_window.py    # Inspection interface
│       ├── simple_inspection_window.py      # Simplified inspection
│       └── advanced_inspection_window.py   # Advanced inspection with threading
└── data/                           # Data storage
    ├── reference_images/           # Reference images for comparison
    ├── mask_images/               # Inspection masks
    ├── processed/                 # Processed inspection data
    ├── raw/                       # Raw captured images
    └── inspection_logs/           # Inspection audit logs
```

## 🛠️ Installation

### Prerequisites
- Python 3.9+ (Anaconda recommended)
- OpenCV compatible camera
- macOS/Linux/Windows

### Setup
1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd AI_VDI_System
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python main.py
   ```

## 📋 Dependencies

```
PyQt5>=5.15.0
opencv-python>=4.5.0
numpy>=1.21.0
Pillow>=8.3.0
```
- Barcode scanner (optional)

## Installation

### 1. Clone or Download the Project
```bash
# If using git
git clone <repository-url>
cd AI_VDI_System

# Or download and extract the zip file
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate virtual environment
# On Windows:
venv\\Scripts\\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the System
Edit `config.ini` to match your hardware setup:
- Set correct camera ID
- Configure model paths
- Adjust detection thresholds
- Enable/disable barcode scanning

## Quick Start

### 1. Run the Application
```bash
python src/main.py
```

### 2. Basic Operation
1. **Start the Application**: The main window will open with camera feed
2. **Configure Settings**: Adjust camera and ML parameters as needed
3. **Load ML Model**: Ensure a trained model is available in `data/models/`
4. **Start Inspection**: Click the start button to begin automated inspection
5. **Monitor Results**: View real-time results in the status panel

## Project Structure

```
AI_VDI_System/
├── src/                      # Source code
│   ├── ui/                   # GUI components
│   │   ├── __init__.py
│   │   ├── main_window.py    # Main application window
│   │   └── widgets.py        # Custom GUI widgets
│   ├── core/                 # Core system logic
│   │   ├── __init__.py
│   │   ├── inspection_manager.py  # Main orchestrator
│   │   ├── camera_handler.py     # Camera operations
│   │   └── barcode_handler.py    # Barcode scanning
│   ├── ml/                   # Machine Learning components
│   │   ├── __init__.py
│   │   ├── model.py          # Model definitions
│   │   ├── inference.py      # Inference engine
│   │   └── preprocess.py     # Image preprocessing
│   └── main.py               # Application entry point
├── data/                     # Data storage
│   ├── raw/                  # Original images
│   ├── processed/            # Processed data
│   ├── models/               # Trained ML models
│   └── inspection_logs/      # Inspection results
├── requirements.txt          # Python dependencies
├── config.ini               # System configuration
└── README.md                # This file
```

## Configuration

### Camera Settings
```ini
[camera]
camera_id = 0                 # Camera device ID
resolution_width = 1920       # Camera resolution
resolution_height = 1080
fps = 30                      # Frames per second
```

### ML Model Settings
```ini
[ml]
model_path = data/models/defect_model.pth  # Path to trained model
model_type = cnn              # Model architecture (cnn, vit)
device = cpu                  # Processing device (cpu, cuda)
threshold = 0.5               # Detection confidence threshold
```

### Barcode Settings
```ini
[barcode]
enabled = true                # Enable barcode scanning
scan_timeout = 5.0           # Scan timeout in seconds
supported_formats = CODE128,QR,EAN13  # Supported formats
```

## Training Your Own Model

### 1. Prepare Training Data
- Organize images in `data/raw/` folder
- Create subfolders for each class (e.g., "good", "defect")
- Ensure sufficient samples for each class

### 2. Model Training (Example)
```python
from src.ml.model import ModelFactory, ModelTrainer
import torch

# Create model
config = {'num_classes': 2, 'pretrained': True}
model = ModelFactory.create_model('cnn', config)

# Initialize trainer
trainer = ModelTrainer(model, device='cpu')

# Train model (implement your training loop)
# trainer.train_epoch(dataloader, optimizer, criterion)

# Save trained model
trainer.save_model('data/models/defect_model.pth')
```

## Usage Examples

### Basic Inspection
```python
from src.core.inspection_manager import InspectionManager

# Load configuration
config = load_config('config.ini')

# Create inspection manager
manager = InspectionManager(config)

# Start inspection
if manager.start_inspection():
    print("Inspection started successfully")
    
    # Get latest result
    result = manager.get_latest_result()
    if result:
        print(f"Status: {result['status']}")
        print(f"Confidence: {result['confidence']:.2f}")
```

### Manual Image Processing
```python
from src.ml.inference import InferenceEngine
import cv2

# Initialize inference engine
config = {'model_path': 'data/models/defect_model.pth'}
engine = InferenceEngine(config)
engine.load_model()

# Process image
image = cv2.imread('test_image.jpg')
result = engine.predict(image)

print(f"Defect detected: {result['is_defective']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## API Reference

### InspectionManager
- `start_inspection()`: Start the inspection process
- `stop_inspection()`: Stop the inspection process
- `get_latest_result()`: Get the most recent inspection result
- `get_inspection_history()`: Get historical inspection data

### CameraHandler
- `initialize()`: Initialize camera connection
- `capture_image()`: Capture a single image
- `capture_continuous()`: Generator for continuous capture
- `cleanup()`: Release camera resources

### InferenceEngine
- `load_model()`: Load trained ML model
- `predict(image)`: Run inference on single image
- `predict_batch(images)`: Run batch inference
- `set_threshold(threshold)`: Set detection threshold

## Troubleshooting

### Common Issues

1. **Camera Not Found**
   - Check camera connection
   - Verify camera_id in config.ini
   - Test with different camera IDs (0, 1, 2, etc.)

2. **Model Loading Error**
   - Ensure model file exists in specified path
   - Check model compatibility with current PyTorch version
   - Verify model_type setting in config

3. **Barcode Scanner Issues**
   - Install pyzbar: `pip install pyzbar`
   - Check if barcode is clearly visible
   - Verify supported_formats in config

4. **Performance Issues**
   - Reduce camera resolution
   - Use GPU if available (set device = cuda)
   - Optimize model architecture

### Logging
Check log files in the `logs/` directory for detailed error information:
- `ai_vdi_system.log`: Main application log
- Console output for real-time debugging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section
- Review log files for error details
- Create an issue in the project repository

## Changelog

### Version 1.0.0
- Initial release
- Basic defect detection functionality
- Camera and barcode integration
- PyQt5 GUI interface
- Configurable system settings