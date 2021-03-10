# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:29:46 2019

@author: udprajapati
"""

# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
import numpy as np
import flask
from flask import current_app, Flask, render_template, request
import io
import pickle
import pandas as pd
from io import StringIO
from pandas.io.json import json_normalize
 
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def load_model():
    global model, ohe_risk_cat, std_scaler, le_enc_tgt
    model = pickle.load(open('model_files/rf_classifier.sav', 'rb'))
    ohe_risk_cat = pickle.load(open('model_files/ohe_risk_cat.sav', 'rb'))
    std_scaler = pickle.load(open('model_files/std_scaler.sav', 'rb'))
    le_enc_tgt = pickle.load(open('model_files/le_enc_tgt.sav', 'rb'))
    
def prepare_data(X_test):
    X_test.columns = ['Account', 'Risk Cat', 'Avg Pay Days', 'Delayed Amt %', 'P30DTOA',
       '90DSO']
    X_test = X_test.drop('Account', axis=1)
    X_test_risk_enc = pd.DataFrame(ohe_risk_cat.transform(X_test[['Risk Cat']]).toarray())
    X_test = X_test.reset_index().drop('index', axis=1)
    X_test = X_test.join(X_test_risk_enc)
    X_test = X_test.drop('Risk Cat', axis=1)
    X_test = std_scaler.transform(X_test)
    return X_test

@app.route("/predict", methods=["GET","POST"])
def predict():
    if flask.request.method == "POST":
        print('1')
        if flask.request.get_json():
            json_data = flask.request.get_json()
            df_test = pd.DataFrame.from_dict(json_data)
            print(df_test)
            df_test1 = df_test.copy()
            df_test = prepare_data(df_test)
            preds = le_enc_tgt.inverse_transform(model.predict(df_test))
            print(preds)
            df_test1['ML predictions'] = preds
            parsed = df_test1.to_json(orient="columns")
            return parsed      
    return 'Hello'

@app.route("/predict_mass", methods=["GET","POST"])
def predict_mass():
    if flask.request.method == "POST":
        if flask.request.files.get("csv_file"):
            file = flask.request.files["csv_file"]
            bytes_data = file.read()
            df_test = pd.read_csv(file.filename)
            df_test1 = df_test.copy()
            df_test = prepare_data(df_test)
            preds = le_enc_tgt.inverse_transform(model.predict(df_test))
            print(preds)
            df_test1['ML predictions'] = preds
            return render_template('form.html',predictions=df_test1.values, flag=True)            
    return render_template('form.html')  
             
if __name__ == "__main__":
	print(("* Loading models and Flask starting server..."
		"please wait until server has fully started"))
	load_model()
	app.run(debug=True)
    #app.run(host='0.0.0.0', port=8080)