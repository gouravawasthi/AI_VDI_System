"""
Image Preprocessing for ML Model Input
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional, Union
import torch
from torchvision import transforms


class ImagePreprocessor:
    """
    Handles image preprocessing for ML model input
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize image preprocessor
        
        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config
        self.target_size = config.get('target_size', (224, 224))
        self.normalize = config.get('normalize', True)
        self.mean = config.get('mean', [0.485, 0.456, 0.406])  # ImageNet defaults
        self.std = config.get('std', [0.229, 0.224, 0.225])    # ImageNet defaults
        self.resize_method = config.get('resize_method', 'bilinear')
        self.crop_center = config.get('crop_center', True)
        
        self.logger = self._setup_logger()
        self._setup_transforms()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for image preprocessor"""
        logger = logging.getLogger('ImagePreprocessor')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_transforms(self):
        """Setup PyTorch transforms for preprocessing"""
        transform_list = []
        
        # Resize
        if self.resize_method == 'bilinear':
            transform_list.append(transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR))
        else:
            transform_list.append(transforms.Resize(self.target_size))
        
        # Center crop if enabled
        if self.crop_center:
            transform_list.append(transforms.CenterCrop(self.target_size))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize if enabled
        if self.normalize:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))
        
        self.transform = transforms.Compose(transform_list)
    
    def preprocess(self, image: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """
        Preprocess a single image for model input
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            
        Returns:
            Preprocessed image as tensor or numpy array
        """
        try:
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Apply transforms
            if self.transform:
                # Convert to PIL Image format expected by torchvision transforms
                from PIL import Image
                pil_image = Image.fromarray(image_rgb.astype(np.uint8))
                processed = self.transform(pil_image)
                return processed
            else:
                # Manual preprocessing
                processed = self._manual_preprocess(image_rgb)
                return processed
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            # Return a default tensor if preprocessing fails
            return torch.zeros((3, *self.target_size))
    
    def _manual_preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Manual preprocessing without PyTorch transforms
        
        Args:
            image: Input image as numpy array (RGB format)
            
        Returns:
            Preprocessed image as numpy array
        """
        # Resize image
        resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to float and normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Apply channel-wise normalization if enabled
        if self.normalize:
            for c in range(3):
                normalized[:, :, c] = (normalized[:, :, c] - self.mean[c]) / self.std[c]
        
        # Convert from HWC to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        return transposed
    
    def preprocess_batch(self, images: list) -> Union[np.ndarray, torch.Tensor]:
        """
        Preprocess a batch of images
        
        Args:
            images: List of input images as numpy arrays
            
        Returns:
            Batch of preprocessed images
        """
        try:
            processed_images = []
            
            for image in images:
                processed = self.preprocess(image)
                processed_images.append(processed)
            
            # Stack into batch
            if isinstance(processed_images[0], torch.Tensor):
                batch = torch.stack(processed_images)
            else:
                batch = np.stack(processed_images)
            
            return batch
            
        except Exception as e:
            self.logger.error(f"Error preprocessing batch: {e}")
            # Return empty batch
            if self.transform:
                return torch.zeros((len(images), 3, *self.target_size))
            else:
                return np.zeros((len(images), 3, *self.target_size))
    
    def resize_image(self, image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            target_size: Target size as (width, height), uses default if None
            
        Returns:
            Resized image
        """
        size = target_size or self.target_size
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image values
        
        Args:
            image: Input image (0-255 or 0-1 range)
            
        Returns:
            Normalized image
        """
        # Convert to float if needed
        if image.dtype != np.float32:
            normalized = image.astype(np.float32)
        else:
            normalized = image.copy()
        
        # Normalize to [0, 1] if needed
        if normalized.max() > 1.0:
            normalized = normalized / 255.0
        
        # Apply channel-wise normalization
        if self.normalize and len(normalized.shape) == 3:
            for c in range(min(3, normalized.shape[2])):
                normalized[:, :, c] = (normalized[:, :, c] - self.mean[c]) / self.std[c]
        
        return normalized
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Denormalize image (reverse normalization)
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image in [0, 255] range
        """
        denormalized = image.copy()
        
        # Reverse channel-wise normalization
        if self.normalize and len(denormalized.shape) == 3:
            for c in range(min(3, denormalized.shape[2])):
                denormalized[:, :, c] = (denormalized[:, :, c] * self.std[c]) + self.mean[c]
        
        # Convert to [0, 255] range
        denormalized = np.clip(denormalized * 255.0, 0, 255).astype(np.uint8)
        
        return denormalized
    
    def augment_image(self, image: np.ndarray, augmentation_type: str = 'random') -> np.ndarray:
        """
        Apply data augmentation to image
        
        Args:
            image: Input image
            augmentation_type: Type of augmentation to apply
            
        Returns:
            Augmented image
        """
        try:
            if augmentation_type == 'flip_horizontal':
                return cv2.flip(image, 1)
            elif augmentation_type == 'flip_vertical':
                return cv2.flip(image, 0)
            elif augmentation_type == 'rotate_90':
                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif augmentation_type == 'rotate_180':
                return cv2.rotate(image, cv2.ROTATE_180)
            elif augmentation_type == 'rotate_270':
                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif augmentation_type == 'brightness':
                alpha = np.random.uniform(0.8, 1.2)  # Brightness factor
                return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            elif augmentation_type == 'contrast':
                alpha = np.random.uniform(0.8, 1.2)  # Contrast factor
                return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
            elif augmentation_type == 'random':
                # Apply random augmentation
                augmentations = ['flip_horizontal', 'rotate_90', 'brightness', 'contrast']
                chosen_aug = np.random.choice(augmentations)
                return self.augment_image(image, chosen_aug)
            else:
                return image
                
        except Exception as e:
            self.logger.error(f"Error applying augmentation: {e}")
            return image
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get preprocessor configuration
        
        Returns:
            Configuration dictionary
        """
        return {
            'target_size': self.target_size,
            'normalize': self.normalize,
            'mean': self.mean,
            'std': self.std,
            'resize_method': self.resize_method,
            'crop_center': self.crop_center
        }