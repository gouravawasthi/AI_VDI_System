"""
ML Inference Engine for Defect Detection
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    from .model import ModelFactory, DefectDetectionCNN
    from .preprocess import ImagePreprocessor
except ImportError:
    # Fallback for direct execution from root
    from src.ml.model import ModelFactory, DefectDetectionCNN
    from src.ml.preprocess import ImagePreprocessor


class InferenceEngine:
    """
    Inference engine for running defect detection predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize inference engine
        
        Args:
            config: Configuration dictionary for ML inference
        """
        self.config = config
        self.model_path = config.get('model_path', 'data/models/defect_model.pth')
        self.model_type = config.get('model_type', 'cnn')
        self.device = config.get('device', 'cpu')
        self.threshold = config.get('threshold', 0.5)
        self.class_names = config.get('class_names', ['Good', 'Defect'])
        
        self.model = None
        self.preprocessor = None
        self.is_loaded = False
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for inference engine"""
        logger = logging.getLogger('InferenceEngine')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_model(self) -> bool:
        """
        Load the trained model for inference
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            model_path = Path(self.model_path)
            
            if not model_path.exists():
                self.logger.warning(f"Model file not found: {model_path}")
                # Create a dummy model for demonstration
                self._create_dummy_model()
                return True
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model instance
            model_config = {
                'num_classes': checkpoint.get('num_classes', 2),
                'pretrained': False  # Don't use pretrained weights when loading checkpoint
            }
            
            self.model = ModelFactory.create_model(self.model_type, model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize preprocessor
            preprocess_config = self.config.get('preprocessing', {})
            self.preprocessor = ImagePreprocessor(preprocess_config)
            
            self.is_loaded = True
            self.logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration purposes"""
        model_config = {
            'num_classes': len(self.class_names),
            'pretrained': False
        }
        
        self.model = ModelFactory.create_model(self.model_type, model_config)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize preprocessor
        preprocess_config = self.config.get('preprocessing', {})
        self.preprocessor = ImagePreprocessor(preprocess_config)
        
        self.is_loaded = True
        self.logger.info("Created dummy model for demonstration")
    
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on an input image
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.is_loaded or self.model is None:
            self.logger.error("Model not loaded")
            return {
                'error': 'Model not loaded',
                'confidence': 0.0,
                'class_id': 0,
                'class_name': 'Unknown'
            }
        
        try:
            # Preprocess image
            processed_image = self.preprocessor.preprocess(image)
            
            # Convert to tensor and add batch dimension
            if isinstance(processed_image, np.ndarray):
                # Convert from numpy to tensor
                if len(processed_image.shape) == 3:
                    # Add batch dimension: (C, H, W) -> (1, C, H, W)
                    input_tensor = torch.from_numpy(processed_image).unsqueeze(0)
                else:
                    input_tensor = torch.from_numpy(processed_image)
            else:
                input_tensor = processed_image.unsqueeze(0)
            
            input_tensor = input_tensor.to(self.device).float()
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                confidence = confidence.item()
                predicted_class = predicted_class.item()
            
            # Prepare result
            result = {
                'confidence': confidence,
                'class_id': predicted_class,
                'class_name': self.class_names[predicted_class] if predicted_class < len(self.class_names) else 'Unknown',
                'probabilities': probabilities.cpu().numpy().tolist()[0],
                'is_defective': predicted_class == 1 and confidence > self.threshold  # Assuming class 1 is 'Defect'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            return {
                'error': str(e),
                'confidence': 0.0,
                'class_id': 0,
                'class_name': 'Error'
            }
    
    def predict_batch(self, images: list) -> list:
        """
        Run inference on a batch of images
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            List of prediction results
        """
        if not self.is_loaded or self.model is None:
            self.logger.error("Model not loaded")
            return []
        
        results = []
        
        try:
            # Preprocess all images
            processed_images = []
            for image in images:
                processed_image = self.preprocessor.preprocess(image)
                if isinstance(processed_image, np.ndarray):
                    processed_image = torch.from_numpy(processed_image)
                processed_images.append(processed_image)
            
            # Stack into batch tensor
            batch_tensor = torch.stack(processed_images).to(self.device).float()
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(probabilities, 1)
            
            # Process results
            for i in range(len(images)):
                confidence = confidences[i].item()
                predicted_class = predicted_classes[i].item()
                
                result = {
                    'confidence': confidence,
                    'class_id': predicted_class,
                    'class_name': self.class_names[predicted_class] if predicted_class < len(self.class_names) else 'Unknown',
                    'probabilities': probabilities[i].cpu().numpy().tolist(),
                    'is_defective': predicted_class == 1 and confidence > self.threshold
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during batch inference: {e}")
            return []
    
    def set_threshold(self, threshold: float):
        """
        Set confidence threshold for defect detection
        
        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.threshold = threshold
            self.logger.info(f"Threshold set to {threshold}")
        else:
            self.logger.warning(f"Invalid threshold value: {threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary containing model information
        """
        if not self.is_loaded or self.model is None:
            return {'error': 'Model not loaded'}
        
        try:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            info = {
                'model_type': self.model_type,
                'model_path': self.model_path,
                'device': self.device,
                'threshold': self.threshold,
                'class_names': self.class_names,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'input_size': getattr(self.preprocessor, 'target_size', 'Unknown')
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Cleanup inference engine resources"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_loaded = False
            self.logger.info("Inference engine cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")