import os
import sys
from flask import *
import torch



app = Flask(__name__)
app.debug = True

root = os.getcwd()
<<<<<<< Updated upstream
from Classifier import CatFaceIdentifier, val_transform
=======

from Classifier import CatFaceIdentifier, temp_transform
>>>>>>> Stashed changes

model = CatFaceIdentifier().cuda()
checkpoint = torch.load(os.path.join(root, '..', "ckpt.pt"))
model.load_state_dict(checkpoint['model_state_dict'])

@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/image_get')
def image_get():
    return render_template('upload.html')


@app.route('/result_post', methods = ['GET', 'POST'])
def result_post():
    if request.method == 'POST':
        user_img = request.files['user_img']
        user_img.save("./flask_deep/images/"+str(user_img.filename))
        #Need to add Preprocess output
        img = temp_transform(user_img)
        #Suggestion: Get Video -> Can make the accs better
        pred = model(img)
    return render_template('result.html', pred = pred)
