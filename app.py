import io
import json
import os

from torchvision import models

import albumentations
from albumentations import * 
from albumentations.pytorch import ToTensorV2

from PIL import Image
from flask import Flask, jsonify, request
from efficientnet_pytorch import EfficientNet
import numpy as np
import pandas as pd
import torch

class cfg:
    name="efficientnet-b1",
    num_classes=2,
    h = 300,
    w = 200,
    mean = (0.5601935833753097, 0.5241012118058834, 0.5014569968941146),
    std = (0.2331860325508089, 0.2430003291910566, 0.24567521565050057)

app = Flask(__name__)
model = EfficientNet.from_pretrained("efficientnet-b1", num_classes=2)
model.load_state_dict(torch.load("./firstmodel.pt",map_location="cpu"))
model.eval()

def transform_image(image_bytes):
    image=Image.open(io.BytesIO(image_bytes))
    image=np.array(image)
    transformations = Compose([
        CenterCrop(height=300,width=200, p=1.0),
        Normalize(mean=cfg.mean, std=cfg.std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0)], 
        p=1.0)
    return transformations(image=image)["image"].unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    pred=model(tensor)
    pred = pred.argmax(dim=-1)
    ans = pred.detach().cpu().numpy()[0]
    return ans

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        label_id = int(get_prediction(image_bytes=img_bytes))
        return jsonify({'label_id': label_id})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')

# py -m flask run
