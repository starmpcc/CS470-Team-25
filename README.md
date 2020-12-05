# CS470-Team-25

Requirements
=====
To install Requirements, use

`$ pip3 install -r requirements.txt`

The model is written and tested on cuda enviroment

Also, you have to download `ckpt.pt` from below link and place into `./cat_id/flask_deep/`
https://drive.google.com/file/d/1pcQRjzuhWySDOAnumXhN16y5s9jBK1Ni/view?usp=sharing


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
The main Function that preprocess and train model

`resave.py`
---
Utility script to reduce the size of the checkpoint file

`test_model.py`
----
Plot train/valid accuracy/loss and show some examples

`./cat_id`
----
Directory that include service frontend/backend
 - `__init__.py`

    Script that execute server

`./data_collect`
----
Directory that include crawler & dataset
 
 - `rename.py`
    
    Utility function to reorganize file structure