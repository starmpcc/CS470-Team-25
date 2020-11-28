import os
import sys
from flask import *
import torch
from flask import Flask, request
from werkzeug.utils import secure_filename
from PIL import Image
root = os.getcwd()

app = Flask(__name__)
app.debug = True

from .Classifier import CatFaceIdentifier, val_transform


'''
Import classifier
TODO : Activate the comments
TODO : In Classifier.py, val_transform must be const.
'''
device = torch.device('cpu')
model = CatFaceIdentifier()
checkpoint = torch.load(root+"/flask_deep/ckpt.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

user_img_src = "images/cat.jpg"

@app.route('/')
def index():
    return render_template('upload.html', user_img = user_img_src)

@app.route('/upload_post')
def upload_post():
    return render_template('upload.html', user_img = user_img_src)

@app.route('/result_post', methods = ['GET', 'POST'])
def result_post():
    pred = None
    top5_list = []
    if request.method == "POST" :
        user_img = request.files['user_img']
        user_img.save(os.path.join(root, "flask_deep", "static", 'images/{}.jpg'.format("temp_img")))
        '''
        classify given image
        TODO : Activate the comments
        '''
        user_img_src = 'images/{}.jpg'.format("temp_img")
        img = Image.open(os.path.join(root, "flask_deep", "static", user_img_src))
        img = val_transform(img).unsqueeze(0).to('cpu')
        res = model(img)
        _, pred = torch.max(res,1)
        _, top5 = torch.topk(res, 5, 1)
        pred = pred.item()
        pred_src = 'categories/cat_'+str(pred)+'.jpg'
        top5_list = []
        top5 = top5[0]
        for item in top5.tolist():
            top5_list.append('categories/cat_'+str(item)+'.jpg')
    #Return predicted image source and top5 images source list
    return render_template('result.html', user_img=user_img_src, pred = pred_src, top5 = top5_list)
