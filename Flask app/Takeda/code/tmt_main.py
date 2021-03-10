from flask import *
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle as pk
from sklearn.metrics import confusion_matrix, accuracy_score,average_precision_score,classification_report,f1_score
#import urllib.parse
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import time
#from flask_restful import Api,Resource
from flask import jsonify
import json
import os,shutil
import zipfile
import glob
import tensorflow as tf
import physicianDeckReview as slideSim
import time
import swifter
from threading import Thread as t
import queue
from pathlib import Path

UPLOAD_FOLDER = './uploads_f/'#'/uploads_f'
ALLOWED_EXTENSIONS = set(['zip'])
pd.set_option('display.max_colwidth', -1)
app = Flask(__name__,template_folder='../templates',static_folder='../static')
#app=Flask(__name__,template_folder='../templates',static_folder='../static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
global recent_upload



@app.route('/home', methods=['GET', 'POST'])

def upload_file():

    if request.method == 'POST':
        print(os.getcwd())
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filelist = [ f for f in os.listdir('../uploads_f/') ]
            for f in filelist:
                shutil.rmtree('../uploads_f/'+f)
            time.sleep(5)
            #os.mkdir('./uploads_f/')
            filename = secure_filename(file.filename)
            file.save('../uploads_f/'+filename)
            with zipfile.ZipFile('../uploads_f/'+filename, 'r') as zipObj:
                zipObj.extractall('../uploads_f/'+filename[:-4] )
            recent_upload=filename[:-4]
            extension = '.zip'
            f_list = [f for f in listdir('../uploads_f/') if isfile(join('../uploads_f/', f))]
            print(f_list)
            os.remove(os.path.join('../uploads_f/', filename))
            return render_template('upload.html',file_list = f_list)
    f_list = [f for f in listdir('../uploads_f/') if isfile(join('../uploads_f/', f))]
    return render_template('upload.html',file_list = f_list)

@app.route('/blog/<int:postID>')
def show_blog(postID):
   return 'Blog Number %d' % postID

@app.route("/preview")
def preview(filename):
    return 'File Uploaded'

@app.route("/predict")
def predict():
    return render_template('predict.html')

@app.route('/function')
def get_ses():
    print("server has called api")
    filelist = [ f for f in os.listdir('../uploads_f/') if f[:-4]!='.zip']
    print(filelist)
    for f in filelist:
            folder_name=f
            print(folder_name)

    m = slideSim.slideSimilarity(folder_name)
    return render_template('upload0.html',tables=[m.to_html(index=False, classes=' table-hover table-condensed  table-striped center')])
if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0',debug=True,port=5000)

'''

@app.route("/predictvalue", methods=['GET', 'POST'])
def variable():
    if request.method == "POST":
        req = request.form
        print(req)
        values=[]
        columnss=[]
        for key, value in req.items():
            print(key, '->', value)
            columnss.append(key)
            values.append(value)
            file = open("models.pkl", "rb")
        print(columnss)
        print(values)
        df = pd.DataFrame([values],columns=columnss)
        print(df)
        trained_encoder = pk.load(file)  #Pickle file first load the OneHotEncoder
        trained_model_for_prediction = pk.load(file)
        label_encoder_yy= pk.load(file)
        y_predict=trained_encoder.transform(df)
        y_pred=trained_model_for_prediction.predict(y_predict)
        y_pred_oo_pred=label_encoder_yy.inverse_transform(y_pred)
        df['MATNR']=y_pred_oo_pred
        print(df)
        file.close()
        dataset1=df
        df.to_csv('./uploads_f/Predict1.csv')
        return render_template('upload0.html',tables=[dataset1.to_html(classes='dataset1')])
        #return 'Check Console'
    else:
        return 'Check with Expert'
@app.route('/predictsap', methods=["GET","POST","PUT"])
def summary():
    if request.method == "POST":
        if request.get_json():
            req = request.get_json()
            dataframe = pd.DataFrame.from_dict(req, orient="index")
            print(dataframe)
            file = open("models.pkl", "rb")
            trained_encoder = pk.load(file)  #Pickle file first load the OneHotEncoder
            trained_model_for_prediction = pk.load(file)
            label_encoder_yy= pk.load(file)
            y_predict=trained_encoder.transform(dataframe)
            y_pred=trained_model_for_prediction.predict(y_predict)
            y_pred_oo_pred=label_encoder_yy.inverse_transform(y_pred)
            file.close()
            dataframe['MATNR']=y_pred_oo_pred
            dataframe=dataframe.to_json(orient='index')
            return dataframe, 200
        return "Thanks! I did not receive json", 200
    #dataset1 = pd.read_csv('./uploads_f/Test.tsv', sep='\t',dtype=object)
    #dataset1 = dataset1.to_json(orient='index')
    #return dataset1
    return "Thanks for sending a request but that is not a post request"
'''
