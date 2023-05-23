#!/bin/bash
set -ex

python3 -m venv nlp_env

source nlp_env/bin/activate >/dev/null 2>&1

pip install -r requirements.txt

python3 train.py

cd flask-app  #TO DO: the train.py needs to modified to add code for saving the model

python3 app.py
