# -*- coding: utf-8 -*-

from flask import Flask, render_template, request

import datetime
import tensorflow as tf
import numpy as np
import pickle
import numpy as np





app = Flask(__name__)


with open('./model/cospi_model.sav', 'rb') as f:
    model = pickle.load(f)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        dat_temp = float(request.form['dat_temp'])
        covid_temp = float(request.form['covid_temp'])
        dong_temp = float(request.form['dong_temp'])
    price = 0

    x_test = np.array([[dat_temp, covid_temp, dong_temp]])
    y_predict = model.predict(x_test)
    price = round(y_predict[0][0],2)
    return render_template('index.html', price = price)


if __name__ == '__main__':
    app.run(debug=True)
