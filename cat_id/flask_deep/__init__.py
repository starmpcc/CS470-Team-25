import os
import sys
from flask import *
import torch
from flask import Flask, request
from werkzeug.utils import secure_filename
from PIL import Image
import random
root = os.getcwd()

app = Flask(__name__)
app.debug = True

from .Classifier import CatFaceIdentifier, val_transform


device = torch.device('cpu')
model = CatFaceIdentifier()
checkpoint = torch.load(root + "/flask_deep/ckpt.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

'''
Import classifier
TODO : Activate the comments
TODO : In Classifier.py, val_transform must be const.
'''

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
        file_name_temp = 'images/{}_{}.jpg'.format("temp_img", random.randint(1, 1000))
        user_img.save(
            os.path.join(root, "flask_deep", "static", file_name_temp))

        user_img_src = file_name_temp  # 'images/{}.jpg'.format("temp_img")
        img = Image.open(os.path.join(root, "flask_deep", "static", user_img_src))
        #img = misc.imread(user_img)
        #img = img[:,:, 3]
        img = val_transform(img).unsqueeze(0).to('cpu')
        #print(type(user_img))
        res = model(img)
        res = torch.nn.functional.softmax(res[0], dim=0)
        _, pred = torch.max(res,0)
        prob5, top5 = torch.topk(res, 5, 0)
        #pred = pred.item()
        pred_src = 'categories/cat_'+str(pred)+'.jpg'
        top5_list = []
        #top5 = (top5[0]).tolist()
        top5 = (top5).tolist()
        #prob5_list = (prob5[0]).tolist()
        prob5_list = ["{0:.2f}%".format(p*100) for p in prob5]
        for item in top5:
            top5_list.append('categories/cat_'+str(item)+'.jpg')
    #Return predicted image source and top5 images source list
    return render_template('result.html', user_img=user_img_src, pred = user_img_src, top5 = top5_list, prob5 = prob5_list)