import shutil

import cv2
from flask import Flask, request, render_template, jsonify, url_for, send_file
from PIL import Image
import torch
import numpy as np
import os
import glob
import random
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

# Initialize Flask
app = Flask(__name__)

uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

# preprocessing pipeline
dinov2_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

# load model
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# load features
all_features_path = 'model/all_features_1.npy'
if os.path.exists(all_features_path):
    all_features = np.load(all_features_path, allow_pickle=True).item()
else:
    raise FileNotFoundError(f"Could not find '{all_features_path}'")

# image directory
image_dir = 'data/pokemon'

# home pg
@app.route('/')
def index():
    return render_template('index.html')

# image upload and return
@app.route('/api/upload', methods=['POST'])
def upload_image():
    uploaded_img = request.files['file']
    if not uploaded_img or uploaded_img.filename == '':
        return jsonify({'error': 'No file uploaded or file name is missing'}), 400

    # Load and preprocess the uploaded image
    uploaded_img_path = os.path.join(uploads_dir, uploaded_img.filename)
    uploaded_img.save(uploaded_img_path)
    input = cv2.imread(uploaded_img_path)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

    input_object_pil = Image.fromarray(np.uint8(input))
    input_object = np.array(input_object_pil)
    input_object_tensor = dinov2_preprocess(input_object_pil)
    input_object_tensor = input_object_tensor.type(torch.float32)

    ## dinov2 feature exatarction
    input_feat = dinov2_model.forward_features(input_object_tensor.unsqueeze(0))
    patch_tokens_input = input_feat['x_norm_patchtokens']  # 1x256x1536
    class_token_input = input_feat['x_norm_clstoken']
    input_feat = class_token_input.detach().cpu().numpy()


    score_highest = 0
    for key in all_features:
        cos_sim = 0
        for idx, feature in enumerate(all_features[key]):
            cos_sim_tmp = cosine_similarity(input_feat.reshape(1, -1), all_features[key][idx].reshape(1, -1))
            cos_sim = cos_sim + cos_sim_tmp
        if cos_sim > score_highest:
            score_highest = cos_sim
            label = key
    print(label)
    file_selected_Pokemon = glob.glob(f'data/pokemon/{label}/*.jpg',
                                      recursive=True)
    random_index_recommend = random.randint(0, len(file_selected_Pokemon) - 1)
    selected_Pokemon_path = file_selected_Pokemon[random_index_recommend]
    selected_Pokemon = cv2.imread(selected_Pokemon_path)
    selected_Pokemon = cv2.cvtColor(selected_Pokemon, cv2.COLOR_BGR2RGB)


    shutil.rmtree(uploads_dir)
    os.makedirs(uploads_dir)

    return jsonify({
        'text': f"Recommended Pokémon is {label}",
        'imageUrl': url_for('get_image', folder=label, filename=os.path.basename(selected_Pokemon_path))
    })
# Endpoint to serve the Pokémon images
@app.route('/images/<folder>/<filename>')
def get_image(folder, filename):
    return send_file(os.path.join(image_dir, folder, filename))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)