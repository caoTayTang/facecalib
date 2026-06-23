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

center = torch.tensor([320., 240., 1.])
# center = torch.tensor([w/2, h/2, 1.])

optim = Optimizer(center, for_inference=True)
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


# ---------------------------------------------------
# Main prediction endpoint
# Accepts image OR video
# ---------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():

    # -------------------------
    # IMAGE
    # -------------------------

    if 'image' in request.files:

        from PIL import Image
        from io import BytesIO

        file = request.files['image']

        img = Image.open(BytesIO(file.read())).convert('RGB')
        img = np.array(img)

        landmarks = extractor.extract_landmarks(img)
        print(f"DEBUG: {landmarks.min()=}, {landmarks.max()=}")
        if landmarks is None:
            return jsonify({'error': 'No face detected'}), 400

        h, w = img.shape[:2]

        # resize landmarks to training resolution
        landmarks[:, 0] *= 640.0 / w
        landmarks[:, 1] *= 480.0 / h

        x = torch.tensor(
            landmarks,
            dtype=torch.float32
        ).unsqueeze(0)
        # (B,68,2) -> (B,2,68)
        x = x.permute(0, 2, 1)
    # -------------------------
    # VIDEO
    # -------------------------

    elif 'video' in request.files:

        file = request.files['video']

        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.mp4') as tmp:

            file.save(tmp.name)
            path = tmp.name

        cap = cv2.VideoCapture(path)

        all_landmarks = []
        frame_id = 0

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            # sample every 5th frame
            if frame_id % 5 != 0:
                frame_id += 1
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            landmarks = extractor.extract_landmarks(frame)

            if landmarks is not None:

                h, w = frame.shape[:2]

                landmarks[:, 0] *= 640.0 / w
                landmarks[:, 1] *= 480.0 / h

                all_landmarks.append(landmarks)

            frame_id += 1

            # # enough frames
            # if len(all_landmarks) >= 50:
            #     break

        cap.release()
        os.remove(path)

        if len(all_landmarks) == 0:
            return jsonify({'error': 'No face detected'}), 400

        x = torch.tensor(
            np.stack(all_landmarks),
            dtype=torch.float32
        )
        x = x.permute(0, 2, 1)
    else:
        return jsonify({
            'error': 'Upload image or video'
        }), 400

    # -------------------------
    # Inference
    # -------------------------

    # when using dualoptimization, we cannot no_grad
    # with torch.no_grad(): 
    K_all = optim.predict_intrinsic(x)
    # average intrinsics over all frames
    # K = K_all.mean(0).unsqueeze(0).repeat(x.shape[0],1,1)
    K = K_all
    S = optim.get_shape(x)
    
    print("x:", x.shape)
    print("S:", S.shape)
    print("K:", K.shape)
    x_epnp = x.permute(0,2,1)
    Xc, R, T = util.EPnP_(x_epnp, S, K)  
    # Perform warm start optimization  
    S, K, R, T = optim.dualoptimization(x, max_iter=10)  
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

    return jsonify({
        'frames_used': int(x.shape[0]),
        'focal_length': float(K[0, 0, 0]),
        'reprojection_error_px': float(reproj_error),
        'intrinsics': K.detach().cpu().numpy().tolist(),
        # one pose per frame
        # 'rotation_matrices': R.cpu().numpy().tolist(),
        # 'translations': T.cpu().numpy().tolist()
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)