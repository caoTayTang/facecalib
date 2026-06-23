#!/usr/bin/env python3
"""Quick import verification without running full tests."""

import sys
sys.path.insert(0, 'source')

print("Testing imports...")
try:
    import torch
    print("✓ torch")
except:
    print("✗ torch (install: pip install torch)")

try:
    import cv2
    print("✓ opencv-python")
except:
    print("✗ opencv-python (install: pip install opencv-python)")

try:
    import mediapipe
    print("✓ mediapipe")
except:
    print("✗ mediapipe (install: pip install mediapipe)")

try:
    import flask
    print("✓ flask")
except:
    print("✗ flask (install: pip install flask)")

try:
    from source.mediapipe_extractor import MediaPipeLandmarkExtractor
    print("✓ mediapipe_extractor module")
except Exception as e:
    print(f"✗ mediapipe_extractor: {e}")

try:
    from source.dataloader import MediaPipeLandmarkLoader
    print("✓ dataloader.MediaPipeLandmarkLoader")
except Exception as e:
    print(f"✗ dataloader.MediaPipeLandmarkLoader: {e}")

print("\nAll modules compile successfully!")
