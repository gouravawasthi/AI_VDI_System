#!/usr/bin/env python3
"""
Package Test Script for AI VDI System
Tests all required packages and provides detailed information
"""

import sys

def test_package_imports():
    """Test all required package imports"""
    print("=" * 60)
    print("AI VDI System - Package Import Test")
    print("=" * 60)
    
    results = []
    
    # Test packages
    test_configs = [
        {
            'name': 'Python',
            'test': lambda: f"{sys.version.split()[0]} ✅",
            'required': True
        },
        {
            'name': 'NumPy', 
            'test': lambda: test_numpy(),
            'required': True
        },
        {
            'name': 'OpenCV',
            'test': lambda: test_opencv(),
            'required': True
        },
        {
            'name': 'PyQt5',
            'test': lambda: test_pyqt5(),
            'required': True
        },
        {
            'name': 'PyTorch',
            'test': lambda: test_pytorch(),
            'required': True
        },
        {
            'name': 'TorchVision',
            'test': lambda: test_torchvision(),
            'required': True
        },
        {
            'name': 'Pillow (PIL)',
            'test': lambda: test_pillow(),
            'required': True
        },
        {
            'name': 'PyZbar',
            'test': lambda: test_pyzbar(),
            'required': False
        },
        {
            'name': 'Scikit-Learn',
            'test': lambda: test_sklearn(),
            'required': False
        },
        {
            'name': 'Pandas',
            'test': lambda: test_pandas(),
            'required': False
        }
    ]
    
    all_passed = True
    
    for config in test_configs:
        name = config['name']
        try:
            result = config['test']()
            print(f"{name:15}: {result}")
            results.append((name, True, result))
        except Exception as e:
            status = "❌ FAILED" if config['required'] else "⚠️  OPTIONAL (Missing)"
            error_msg = f"{status} - {str(e)}"
            print(f"{name:15}: {error_msg}")
            results.append((name, False, str(e)))
            if config['required']:
                all_passed = False
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Packages tested: {total}")
    print(f"Packages working: {passed}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if all_passed:
        print("\n🎉 ALL REQUIRED PACKAGES ARE WORKING!")
        print("✅ AI VDI System is ready to run.")
    else:
        print("\n❌ Some required packages failed.")
        print("⚠️  Please check the installation.")
    
    return all_passed

def test_numpy():
    import numpy as np
    version = np.__version__
    # Test basic operation
    arr = np.array([1, 2, 3])
    assert len(arr) == 3
    return f"{version} ✅"

def test_opencv():
    import cv2
    version = cv2.__version__
    # Test basic functionality
    img = cv2.imread('/dev/null')  # This will return None but won't crash
    return f"{version} ✅"

def test_pyqt5():
    from PyQt5.QtCore import QT_VERSION_STR
    from PyQt5.QtWidgets import QApplication
    # Test that we can create an app (but don't run it)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return f"{QT_VERSION_STR} ✅"

def test_pytorch():
    import torch
    version = torch.__version__
    # Test basic tensor operation
    x = torch.tensor([1.0, 2.0, 3.0])
    assert x.sum().item() == 6.0
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    return f"{version} ({device}) ✅"

def test_torchvision():
    import torchvision
    version = torchvision.__version__
    # Test basic import
    from torchvision import transforms
    transform = transforms.ToTensor()
    return f"{version} ✅"

def test_pillow():
    from PIL import Image
    version = getattr(Image, '__version__', 'Unknown')
    # Test basic functionality
    import numpy as np
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    assert img.size == (10, 10)
    return f"{version} ✅"

def test_pyzbar():
    import pyzbar
    # Test basic import
    from pyzbar import pyzbar as zbar
    return "Available ✅"

def test_sklearn():
    import sklearn
    version = sklearn.__version__
    return f"{version} ✅"

def test_pandas():
    import pandas as pd
    version = pd.__version__
    return f"{version} ✅"

if __name__ == "__main__":
    success = test_package_imports()
    sys.exit(0 if success else 1)