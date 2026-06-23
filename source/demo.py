#!/usr/bin/env python3
"""
Demo script: Create test face image, extract landmarks, and plot them.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch
from face_align import FaceAlignmentExtractor


def create_simple_face_image(size=640, save_path="test_face.jpg"):
    """Create a simple synthetic face image for testing."""
    img = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img)
    
    center_x, center_y = size // 2, size // 2
    
    draw.ellipse(
        [center_x - 100, center_y - 120, center_x + 100, center_y + 100],
        outline='black', width=3
    )
    
    draw.ellipse(
        [center_x - 40, center_y - 50, center_x - 20, center_y - 20],
        fill='black'
    )
    
    draw.ellipse(
        [center_x + 20, center_y - 50, center_x + 40, center_y - 20],
        fill='black'
    )
    
    draw.polygon(
        [(center_x - 10, center_y + 20), (center_x + 10, center_y + 20), (center_x, center_y + 40)],
        fill='black'
    )
    
    draw.line(
        [(center_x - 50, center_y + 60), (center_x + 50, center_y + 60)],
        fill='black', width=2
    )
    
    img.save(save_path)
    print(f"✓ Created test image: {save_path}")
    return save_path


def extract_and_visualize(image_path, device='cpu'):
    """Extract landmarks and create visualization."""
    print(f"\n{'='*70}")
    print(f"Processing: {image_path}")
    print(f"{'='*70}")
    
    from skimage import io
    
    image = io.imread(image_path)
    print(f"Image shape: {image.shape}")
    
    print(f"\nInitializing face_alignment extractor (device={device})...")
    extractor = FaceAlignmentExtractor(device=device)
    
    print(f"Extracting landmarks...")
    landmarks = extractor.extract_landmarks(image)
    
    if landmarks is None:
        print("✗ No face detected!")
        return False
    
    print(f"✓ Extracted {len(landmarks)} landmarks")
    print(f"  Shape: {landmarks.shape}")
    print(f"  Type: {landmarks.dtype}")
    print(f"  X range: [{landmarks[:, 0].min():.1f}, {landmarks[:, 0].max():.1f}]")
    print(f"  Y range: [{landmarks[:, 1].min():.1f}, {landmarks[:, 1].max():.1f}]")
    
    print(f"\nFirst 5 landmarks:")
    for i in range(min(5, len(landmarks))):
        print(f"  [{i}] x={landmarks[i, 0]:.1f}, y={landmarks[i, 1]:.1f}")
    
    print(f"\nCreating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axes[0]
    ax.imshow(image)
    ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=20, alpha=0.8)
    ax.set_title(f'Face Landmarks (68 points)\n{os.path.basename(image_path)}')
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(image)
    for i, (x, y) in enumerate(landmarks):
        ax.scatter([x], [y], c='red', s=30, alpha=0.8)
        if i % 5 == 0:
            ax.text(x + 5, y + 5, str(i), fontsize=8, color='yellow')
    ax.set_title('Landmarks with indices (every 5th labeled)')
    ax.axis('off')
    
    plt.tight_layout()
    
    output_path = image_path.replace('.jpg', '_landmarks.png').replace('.png', '_landmarks.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract face landmarks demo')
    parser.add_argument('--image', default=None, help='Path to image file')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--show', action='store_true', help='Show plot')
    
    args = parser.parse_args()
    
    image_path = args.image
    
    if image_path is None or not os.path.exists(image_path):
        print("Creating synthetic test face image...")
        image_path = create_simple_face_image()
    
    success = extract_and_visualize(image_path, device=args.device)
    
    if success and args.show:
        print("\nDisplaying plot...")
        plt.show()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
