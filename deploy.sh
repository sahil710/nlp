#!/bin/bash
set -ex

git clone https://github.com/sahil710/nlp.git

cd nlp

python3 -m venv nlp_env

source nlp_env/bin/activate >/dev/null 2>&1

pip install -r requirements.txt

python3 train.py

cd flask-app

python3 app.py

