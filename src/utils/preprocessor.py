"""Preprocessor utilities for image data"""
import numpy as np
from PIL import Image
import cv2

def preprocess_input(x, v2=True):
    """Preprocess input images for neural network
    
    Args:
        x: Input image array
        v2: Whether to use v2 preprocessing (mean subtraction)
    
    Returns:
        Preprocessed image array
    """
    x = np.asarray(x, dtype='float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def _imread(image_path):
    """Read image from path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image array in RGB format
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)

def _imresize(image_array, size):
    """Resize image array
    
    Args:
        image_array: Input image array
        size: Target size as tuple (height, width)
        
    Returns:
        Resized image array
    """
    return cv2.resize(image_array, (size[1], size[0]))

def to_categorical(y, num_classes=None):
    """Convert class vector to binary class matrix
    
    Args:
        y: Class vector to convert
        num_classes: Total number of classes
        
    Returns:
        Binary matrix representation of input
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype='float32')
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical