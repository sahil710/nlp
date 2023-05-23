import os
from datetime import datetime
import csv
import pickle5 as pickle
from nltk.tokenize import sent_tokenize
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_FILE_NAME = "nlp_model.pkl"
LOG_FILE_NAME = "log.csv"
LOG_FILE_HEADER = ["time", "input", "result", "runtime"]


def add_to_log_file(row_data):
    with open(LOG_FILE_NAME, "a+") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FILE_HEADER)
        writer.writerow(row_data)


@app.route('/')
def index():
    return render_template("index.htm")


@app.route('/result')
def result():
    input_data = request.args.to_dict().get('input-text')
    print(input_data)

    start_time = datetime.now()
    input_data_tokenized = sent_tokenize(input_data)
    model = pickle.load(open(MODEL_FILE_NAME, 'rb'))
    result = model.predict(input_data_tokenized)
    end_time = datetime.now()

    add_to_log_file({"time": start_time.strftime("%m-%d-%Y %H:%M:%S"), "input": input_data,
                    "result": result[0], "runtime": str(end_time-start_time)})

    return render_template("result.htm", prediction=result[0])


if __name__ == '__main__':
    log_file_exists = os.path.isfile(LOG_FILE_NAME)
    if not log_file_exists:
        with open(LOG_FILE_NAME, "w") as log_file:
            csv.DictWriter(log_file, fieldnames=LOG_FILE_HEADER).writeheader()

    app.run(host='0.0.0.0', port=8080, debug=True)
