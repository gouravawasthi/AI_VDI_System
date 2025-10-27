"""
Machine Learning module for AI VDI System
Contains ML model and inference logic
"""

from .inference import InferenceEngine
from .model import DefectDetectionCNN, ModelFactory
from .preprocess import ImagePreprocessor