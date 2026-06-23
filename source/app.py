import sys
import types
sys.modules['pptk'] = types.ModuleType('pptk')

from flask import Flask, request, jsonify
from flask_cloudflared import run_with_cloudflared

import torch
import cv2
import tempfile
import os
import numpy as np
import util
import losses

from optimizer import Optimizer
from face_align import FaceAlignmentExtractor

app = Flask(__name__)
run_with_cloudflared(app)

# ---------------------------------------------------
# Load model once
# ---------------------------------------------------

optim = Optimizer(center=torch.tensor([0., 0., 1.]), for_inference=True)
optim.load('00_')
optim.set_eval()

extractor = FaceAlignmentExtractor(
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# ---------------------------------------------------
# Health endpoint
# ---------------------------------------------------

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


def normalize_landmarks(landmarks, w, h):
    u = landmarks[:, 0]
    v = landmarks[:, 1]

    cx, cy = w * 0.5, h * 0.5
    fx = fy = max(w, h)

    x = (u - cx) / fx
    y = (v - cy) / fy

    return np.stack([x, y], axis=1)


# ---------------------------------------------------
# Main prediction endpoint
# ---------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():

    img = None
    w, h = None, None

    # -------------------------
    # IMAGE
    # -------------------------
    if 'image' in request.files:

        from PIL import Image
        from io import BytesIO

        file = request.files['image']

        img = Image.open(BytesIO(file.read())).convert('RGB')
        img = np.array(img)
        h, w = img.shape[:2]

        landmarks = extractor.extract_landmarks(img)
        if landmarks is None:
            return jsonify({'error': 'No face detected'}), 400

        landmarks = normalize_landmarks(landmarks, w, h)

        x = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
        x = x.permute(0, 2, 1)

    # -------------------------
    # VIDEO
    # -------------------------
    elif 'video' in request.files:

        file = request.files['video']

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            file.save(tmp.name)
            path = tmp.name

        cap = cv2.VideoCapture(path)

        all_landmarks = []
        frame_id = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_id % 5 != 0:
                frame_id += 1
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            landmarks = extractor.extract_landmarks(frame)

            if landmarks is not None:
                landmarks = normalize_landmarks(landmarks, w, h)
                all_landmarks.append(landmarks)

            frame_id += 1

        cap.release()
        os.remove(path)

        if len(all_landmarks) == 0:
            return jsonify({'error': 'No face detected'}), 400

        x = torch.tensor(np.stack(all_landmarks), dtype=torch.float32)
        x = x.permute(0, 2, 1)

    else:
        return jsonify({'error': 'Upload image or video'}), 400

    # -------------------------
    # Inference
    # -------------------------

    print(f"DEBUG before: {landmarks.min()=}, {landmarks.max()=}")

    B = x.shape[0]
    K = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(x.device)

    S = optim.get_shape(x)

    print("x:", x.shape)
    print("S:", S.shape)
    print("K:", K.shape)

    x_epnp = x.permute(0, 2, 1)

    Xc, R, T = util.EPnP_(x_epnp, S, K)

    S, K, R, T = optim.dualoptimization(x, max_iter=5)

    reproj_error = losses.getError(
        x,
        S,
        R,
        T,
        K,
        show=False,
        loss='l2'
    ).mean()

    print("Mean reproj error:", reproj_error.item(), "pixels")
    print("landmarks range after:", landmarks.min(), landmarks.max())

    return jsonify({
        'frames_used': int(x.shape[0]),
        'focal_length': float(K[0, 0, 0]),
        'reprojection_error_px': float(reproj_error),
        'intrinsics': K.detach().cpu().numpy().tolist(),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)