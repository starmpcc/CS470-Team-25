import os
import sys
from flask import *
import torch
app = Flask(__name__)
app.debug = True

root = os.getcwd()
from Classifier import ACNN, temp_transform

model = ACNN().cuda()
checkpoint = torch.load(os.path.join(root, '..', "ckpt.pt"))
model.load_state_dict(checkpoint['model_state_dict'])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/identifier_get')
def identifier_get():
    return render_template('identifier_get.html')

@app.route('/identifier_post', methods = ['GET', 'POST'])
def identifier_post():
    if request.method == 'POST':
        user_inp = request.files['user_img']
        img = temp_transform(user_inp)
        #Need to add Preprocess output
        #Suggestion: Get Video -> Can make the accs better
        pred = model(img)
        return render_template('identifier_post.html', pred = pred)