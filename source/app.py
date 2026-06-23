import sys
import types
sys.modules['pptk'] = types.ModuleType('pptk')

from flask import Flask, request, jsonify
from flask_cloudflared import run_with_cloudflared
import torch
from optimizer import Optimizer
from face_align import FaceAlignmentExtractor
import numpy as np
from io import BytesIO
from PIL import Image

app = Flask(__name__)
run_with_cloudflared(app)

center = torch.tensor([960/2, 720/2, 1])
optim = Optimizer(center, for_inference=True)
optim.load('00_')
optim.calib_net.eval()
optim.sfm_net.eval()

extractor = FaceAlignmentExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'FaceCalib'})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict intrinsics from landmarks."""
    data = request.json
    x = torch.tensor(data['landmarks']).float().unsqueeze(0)
    with torch.no_grad():
        K = optim.predict_intrinsic(x)
    f = K[0, 0, 0].item()
    return jsonify({
        'focal_length': f,
        'intrinsics': K.detach().cpu().numpy().tolist()
    })

@app.route('/extract', methods=['POST'])
def extract():
    """Extract 68 landmarks from image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    img = Image.open(BytesIO(file.read())).convert('RGB')
    img_array = np.array(img)
    
    landmarks = extractor.extract_landmarks(img_array)
    if landmarks is None:
        return jsonify({'error': 'No face detected'}), 400
    
    return jsonify({'landmarks': landmarks.tolist()})

@app.route('/full', methods=['POST'])
def full_predict():
    """Extract landmarks and predict intrinsics."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    
    file = request.files['image']
    img = Image.open(BytesIO(file.read())).convert('RGB')
    img_array = np.array(img)
    
    landmarks = extractor.extract_landmarks(img_array)
    if landmarks is None:
        return jsonify({'error': 'No face detected'}), 400
    
    x = torch.tensor(landmarks).float().unsqueeze(0)
    with torch.no_grad():
        K = optim.predict_intrinsic(x)
    f = K[0, 0, 0].item()
    
    return jsonify({
        'landmarks': landmarks.tolist(),
        'focal_length': f,
        'intrinsics': K.detach().cpu().numpy().tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
