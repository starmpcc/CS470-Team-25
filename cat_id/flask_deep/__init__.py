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
model = CatFaceIdentifier().cuda()
checkpoint = torch.load(root+"/flask_deep/ckpt.pt")
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
    pred = -1
    if request.method == "POST" :
        user_img = request.files['user_img']
        user_img.save(os.path.join(root, "flask_deep", "static", 'images/{}'.format(secure_filename(user_img.filename))))
        '''
        classify given image
        TODO : Activate the comments
        '''
        user_img_src = 'images/{}'.format(secure_filename(user_img.filename))
        img = Image.open(os.path.join(root, "flask_deep", "static", user_img_src))
        img = val_transform(img).unsqueeze(0).to('cuda')
        res = model(img)
        _, pred = torch.max(res,1)
        _, top5 = torch.topk(res, 5, 1)
        pred = pred.item()
        pred_src = os.path.join(root, "flask_deep", "static", "categories", 'cat_'+str(pred)+'.jpg')
        top5_list = []
        for i in top5.tolist():
            top5_list.append(root, "flask_deep", "static", "categories", 'cat_'+str(i)+'.jpg')
    #Return predicted image source and top5 images source list
    return render_template('result.html', user_img=user_img_src, pred = pred_src, top5 = top5_list)
