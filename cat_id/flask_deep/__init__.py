import os
import sys
from flask import *
import torch
from flask import Flask, request
from werkzeug.utils import secure_filename

root = os.getcwd()

app = Flask(__name__)
app.debug = True

from .Classifier import CatFaceIdentifier, val_transform


'''
Import classifier
TODO : Activate the comments
TODO : In Classifier.py, val_transform must be const.
'''
#model = CatFaceIdentifier().cuda()
#checkpoint = torch.load(root+"/flask_deep/ckpt.pt")
#model.load_state_dict(checkpoint['model_state_dict'])

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
        #img = val_transform(user_img)
        #pred = model(img)
    return render_template('result.html', user_img=user_img_src, pred = pred)
