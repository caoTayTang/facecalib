# Face Alignment Landmark Extraction

Simple face landmark extraction (68 points) using the `face_alignment` library.

## Installation

```bash
pip install face-alignment torch torchvision matplotlib scikit-image
```

## Quick Usage

### Extract & Plot Single Image
```bash
python source/extract_landmarks.py test_image.jpg -o output/
```

### Extract & Plot Batch (Directory)
```bash
python source/extract_landmarks.py ./face_images/ -o output/
```

### Extract & Display
```bash
python source/extract_landmarks.py test_image.jpg --show
```

## Python API

### Extract Landmarks from Single Image
```python
from face_align import FaceAlignmentExtractor

extractor = FaceAlignmentExtractor(device='cpu')
landmarks = extractor.extract_landmarks('image.jpg')  # (68, 2) array
print(landmarks)
```

### Batch Processing
```python
from dataloader import FaceAlignmentLoader
from torch.utils.data import DataLoader

loader = FaceAlignmentLoader(image_dir='./faces/')
batch_loader = DataLoader(loader, batch_size=4)

for batch in batch_loader:
    landmarks = batch['x_img']  # (batch_size, 68, 2)
    print(landmarks.shape)
```

## Files

- **face_align.py** - FaceAlignmentExtractor class
- **extract_landmarks.py** - CLI script for extraction & plotting
- **dataloader.py** - FaceAlignmentLoader (PyTorch Dataset)
- **app.py** - Flask API (original /predict endpoint)

## Examples

Extract landmarks from single image:
```bash
python source/extract_landmarks.py ../test/assets/aflw-test.jpg -o ./output/
```

Batch process directory with GPU:
```bash
python source/extract_landmarks.py ./faces/ -o ./results/ --device cuda
```

Show plots interactively:
```bash
python source/extract_landmarks.py image.jpg --show
```

## Output

The landmark plots show all 68 face landmarks numbered on the image:
- Point indices: 0-68
- Red dots: landmark locations
- Saved as PNG with transparency

## Notes

- Landmarks are in image pixel coordinates (x, y)
- Returns None if no face detected
- Use GPU with `--device cuda` for faster processing
- Batch processing recommended for multiple images
