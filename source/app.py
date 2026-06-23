import sys
import types
# Stub out pptk so util.py's import succeeds without the real package
sys.modules['pptk'] = types.ModuleType('pptk')
sys.path.append('/kaggle/working/FaceCalibration/source')

from flask import Flask, request, jsonify
import torch
from optimizer import Optimizer

app = Flask(__name__)

center = torch.tensor([960/2, 720/2, 1])
optim = Optimizer(center)
optim.load('00_')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    x = torch.tensor(data['landmarks']).float().unsqueeze(0)  # [1, N, 2]

    K = optim.predict_intrinsic(x)
    f = K[0, 0, 0].item()

    return jsonify({
        'focal_length': f,
        'intrinsics': K.detach().cpu().numpy().tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
