from flask import Flask, request
import os
from flask_cors import CORS
import torch
import subprocess
app = Flask(__name__)
CORS(app)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return {'error': 'No image part'}, 400
    file = request.files['image']
    if file.filename == '':
        return {'error': 'No selected file'}, 400

    save_path = '/Users/karthicks7/Downloads/Chess_v1_v2/LICH/ultralytics/test2.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)
    
    print(f"Image saved to {save_path}")
    subprocess.run(['python', '/Users/karthicks7/Downloads/Chess_v1_v2/LICH/ultralytics/ka.py']) 
    if torch.cuda.is_available():
        print("CUDA is available. Running on GPU.")
    else:
        print("CUDA is not available. Running on CPU.")
    return {'message': 'Image saved successfully and ka.py called'}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)