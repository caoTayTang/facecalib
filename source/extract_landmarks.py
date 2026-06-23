#!/usr/bin/env python3
"""
Simple script to extract 68 face landmarks and plot them.
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path
from face_align import FaceAlignmentExtractor


def plot_landmarks(image, landmarks, title="Face Landmarks", save_path=None):
    """Plot image with landmarks overlaid."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    ax.imshow(image)
    if landmarks is not None:
        ax.scatter(landmarks[:, 0], landmarks[:, 1], c='red', s=10, alpha=0.8)
        
        for i, (x, y) in enumerate(landmarks):
            ax.text(x, y, str(i), fontsize=4, color='white')
    
    ax.set_title(title)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.tight_layout()
    return fig


def extract_and_plot_single(image_path, output_dir=None, device='cpu'):
    """Extract landmarks from single image and plot."""
    print(f"\nProcessing: {image_path}")
    
    extractor = FaceAlignmentExtractor(device=device)
    image = io.imread(image_path)
    
    print(f"  Image shape: {image.shape}")
    
    landmarks = extractor.extract_landmarks(image)
    
    if landmarks is None:
        print("  ✗ No face detected")
        return None
    
    print(f"  ✓ Extracted {len(landmarks)} landmarks")
    print(f"    Landmark shape: {landmarks.shape}")
    print(f"    Range X: [{landmarks[:, 0].min():.1f}, {landmarks[:, 0].max():.1f}]")
    print(f"    Range Y: [{landmarks[:, 1].min():.1f}, {landmarks[:, 1].max():.1f}]")
    
    output_path = None
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        output_path = f"{output_dir}/{stem}_landmarks.png"
    
    plot_landmarks(image, landmarks, title=Path(image_path).name, save_path=output_path)
    
    return landmarks


def extract_and_plot_batch(image_dir, output_dir=None, device='cpu'):
    """Extract landmarks from all images in directory."""
    from dataloader import FaceAlignmentLoader
    from torch.utils.data import DataLoader
    
    print(f"\nProcessing images from: {image_dir}")
    
    loader = FaceAlignmentLoader(image_dir=image_dir, device=device)
    print(f"Found {len(loader)} images")
    
    batch_loader = DataLoader(loader, batch_size=1, shuffle=False, num_workers=0)
    
    for batch_idx, batch in enumerate(batch_loader):
        fname = batch['fname'][0]
        landmarks = batch['x_img'][0].numpy()
        image = batch['image'][0].numpy()
        
        if image.dtype == np.float32 and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        print(f"  [{batch_idx + 1}/{len(loader)}] {Path(fname).name} - {len(landmarks)} landmarks")
        
        output_path = None
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            stem = Path(fname).stem
            output_path = f"{output_dir}/{stem}_landmarks.png"
        
        plot_landmarks(image, landmarks, title=Path(fname).name, save_path=output_path)
    
    print(f"✓ Processed {len(loader)} images")


def main():
    parser = argparse.ArgumentParser(
        description='Extract 68 face landmarks and plot them'
    )
    parser.add_argument(
        'input',
        help='Image file or directory of images'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output directory for saved plots'
    )
    parser.add_argument(
        '-d', '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use (cpu or cuda)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show plots (matplotlib interactive)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Face Landmark Extraction & Visualization")
    print("=" * 70)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        extract_and_plot_single(str(input_path), args.output, args.device)
    elif input_path.is_dir():
        extract_and_plot_batch(str(input_path), args.output, args.device)
    else:
        print(f"✗ Path not found: {args.input}")
        return 1
    
    if args.show:
        plt.show()
    elif args.output:
        print(f"\n✓ Plots saved to {args.output}/")
    
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
