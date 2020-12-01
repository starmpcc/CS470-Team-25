# CS470-Team-25

Requirements
=====
To install Requirements, use

`$ pip3 install -r requirements.txt`

The model is written and tested on cuda enviroment

Also, you have to download and place `ckpt.pt` into `./cat_id/flask_deep/`

Experiment
=====
To launch demos, move to `cat_id` directory and execute

`$ python3 server.py`

Then you can access to the demo by move to the address in your browser,

`127.0.0.1:5000`

File Structure
=====
```
.
|-- data_collect
|   |-- crawling.py
|   |-- rename.py
|-- cat_id
|   |-- server.py
|   |-- flask_deep
|       |-- static
|       |   |-- css
|       |-- templates
|       |   |-- result.html
|       |   |-- upload.html
|       |-- init.py
|       |-- Classifier.py
|-- Classifier.py
|-- resave.py
|-- test_model.py
|-- 대표이미지
|-- cropped_cat
|-- cropped_cat_2
```

`Classifier.py`
----
Descrption....


