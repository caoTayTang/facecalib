#!/usr/bin/env python3
"""
Quick-start example: Extract landmarks and predict intrinsics from an image.
"""

import sys
import argparse
import numpy as np
import torch
import json
from pathlib import Path


def example_standalone_extraction():
    """Example 1: Extract landmarks from a single image."""
    print("\n" + "=" * 60)
    print("Example 1: Extract landmarks from image")
    print("=" * 60)
    
    from mediapipe_extractor import extract_landmarks_from_image
    
    image_path = "face_image.jpg"
    try:
        landmarks = extract_landmarks_from_image(image_path)
        if landmarks is not None:
            print(f"✓ Extracted {len(landmarks)} landmarks from {image_path}")
            print(f"  Shape: {landmarks.shape}")
            print(f"  First 3 landmarks (x, y): {landmarks[:3]}")
        else:
            print(f"✗ No face detected in {image_path}")
    except FileNotFoundError:
        print(f"✗ Image not found: {image_path}")


def example_batch_processing():
    """Example 2: Process multiple images using dataloader."""
    print("\n" + "=" * 60)
    print("Example 2: Batch processing with dataloader")
    print("=" * 60)
    
    from dataloader import MediaPipeLandmarkLoader
    from torch.utils.data import DataLoader
    
    image_dir = "./face_images"
    
    try:
        loader = MediaPipeLandmarkLoader(image_dir=image_dir)
        batch_loader = DataLoader(loader, batch_size=4, num_workers=0)
        
        print(f"✓ Created dataloader for {len(loader)} images")
        
        for batch_idx, batch in enumerate(batch_loader):
            landmarks = batch['x_img']
            print(f"  Batch {batch_idx}: {landmarks.shape}")
            
            if batch_idx >= 2:
                break
    except ValueError as e:
        print(f"⊘ {e}")
        print(f"  Create {image_dir} directory and add face images to test")


def example_predict_intrinsics():
    """Example 3: Predict camera intrinsics from landmarks."""
    print("\n" + "=" * 60)
    print("Example 3: Predict camera intrinsics")
    print("=" * 60)
    
    from mediapipe_extractor import MediaPipeLandmarkExtractor
    from optimizer import Optimizer
    
    try:
        extractor = MediaPipeLandmarkExtractor()
        
        center = torch.tensor([960/2, 720/2, 1])
        optim = Optimizer(center)
        optim.load('00_')
        
        image_path = "face_image.jpg"
        landmarks = extractor.extract_landmarks(image_path)
        
        if landmarks is not None:
            x = torch.tensor(landmarks).float().unsqueeze(0)
            K = optim.predict_intrinsic(x)
            f = K[0, 0, 0].item()
            
            print(f"✓ Predicted camera intrinsics from {image_path}")
            print(f"  Focal length: {f:.2f}")
            print(f"  Intrinsic matrix:")
            print(f"  {K[0].numpy()}")
        else:
            print(f"✗ No face detected in {image_path}")
    except FileNotFoundError as e:
        print(f"⊘ Could not load model: {e}")
    except Exception as e:
        print(f"⊘ {e}")


def example_api_client():
    """Example 4: Use the Flask API."""
    print("\n" + "=" * 60)
    print("Example 4: API client (requires running server)")
    print("=" * 60)
    
    example_code = '''
import requests
import base64

# Ensure app.py is running: python app.py

# Example 1: Check server health
response = requests.get('http://localhost:5000/health')
print(response.json())

# Example 2: Extract landmarks (file upload)
with open('face_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/landmarks', files=files)
    result = response.json()
    print(f"Landmarks: {len(result['landmarks'])} points")

# Example 3: End-to-end (image -> landmarks -> intrinsics)
with open('face_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:5000/full-predict', files=files)
    result = response.json()
    print(f"Focal Length: {result['focal_length']:.2f}")
    print(f"Intrinsics: {result['intrinsics']}")

# Example 4: Using base64 encoding
with open('face_image.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {"image": f"data:image/jpeg;base64,{img_b64}"}
response = requests.post('http://localhost:5000/landmarks', json=payload)
print(response.json())
'''
    print("Save this as client.py and run it with the server running:\n")
    print(example_code)


def main():
    parser = argparse.ArgumentParser(
        description='FaceCalib MediaPipe Integration Examples'
    )
    parser.add_argument(
        '--example',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help='Which example to run (0=all, 1-4=specific)'
    )
    parser.add_argument(
        '--image',
        default='face_image.jpg',
        help='Path to test face image'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("FaceCalib MediaPipe Integration - Quick Start Examples")
    print("=" * 60)
    
    examples = [
        example_standalone_extraction,
        example_batch_processing,
        example_predict_intrinsics,
        example_api_client,
    ]
    
    if args.example == 0:
        for example_func in examples:
            try:
                example_func()
            except Exception as e:
                print(f"✗ Example failed: {e}")
    else:
        try:
            examples[args.example - 1]()
        except Exception as e:
            print(f"✗ Example failed: {e}")
    
    print("\n" + "=" * 60)
    print("For more information, see MEDIAPIPE_INTEGRATION.md")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
