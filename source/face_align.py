"""
Simple face landmark extraction using face_alignment library.
"""

import face_alignment
import numpy as np
from skimage import io
import torch


class FaceAlignmentExtractor:
    """Extract 68 face landmarks using face_alignment library."""
    
    def __init__(self, device='cpu'):
        """
        Initialize face_alignment.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=device
        )
    
    def extract_landmarks(self, image_path_or_array):
        """
        Extract 68 face landmarks.
        
        Args:
            image_path_or_array: Path to image or numpy array (H, W, 3)
        
        Returns:
            landmarks: numpy array (68, 2) with (x, y) coordinates
                      or None if no face detected
        """
        if isinstance(image_path_or_array, str):
            image = io.imread(image_path_or_array)
        else:
            image = image_path_or_array
        
        preds = self.fa.get_landmarks(image)
        
        if preds is None or len(preds) == 0:
            return None
        
        landmarks = preds[0]
        
        return landmarks.astype(np.float32)
