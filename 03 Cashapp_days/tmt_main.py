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
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
from flask_restful import Api,Resource
from flask import jsonify
import json

UPLOAD_FOLDER = './uploads_f/'#'/uploads_f'
ALLOWED_EXTENSIONS = set(['tsv','csv'])
pd.set_option('display.max_colwidth', -1)
app = Flask(__name__)
api=Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predictsap', methods=["GET","POST","PUT"])
def summary():
    if request.method == "POST":
        if request.get_json():
            req = request.get_json()
            dataframe = pd.DataFrame.from_dict(req, orient="index")
            print(dataframe)
            dataframe['Debit_CreditInd.']=dataframe['Debit_CreditInd.'].apply(lambda x: 1 if x == 'S' else 0)
            file = open("models.pkl", "rb")
            trained_encoder = pk.load(file)
            trained_model_for_prediction = pk.load(file)
            y_predict=trained_encoder.transform(dataframe)
            y_pred=trained_model_for_prediction.predict(y_predict)
            file.close()
            dataframe['Percent_of_Pay_OR_Amount']=y_pred
            dataframe=dataframe.to_json(orient='index')
            return dataframe, 200
        return "Thanks! I did not receive json", 200
    return "Thanks for sending a request but that is not a post request"

@app.route('/home', methods=['GET', 'POST'])

def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save('./uploads_f/'+filename)
            f_list = [f for f in listdir('./uploads_f/') if isfile(join('./uploads_f/', f))]
            return render_template('upload1.html',file_list = f_list)
            #file.save('./uploads_f/'+filename)
            #return 'File Uploaded'
            #return redirect(url_for('preview'))	<img src='/static/image.jpeg' style = 'margin:auto; width:70%'></img>
            #                        ,filename=filename))
    #f_list = [f for f in listdir('./') if isfile(join('./', f))]
    f_list = [f for f in listdir('./uploads_f/') if isfile(join('./uploads_f/', f))]
    return render_template('upload.html',file_list = f_list)

@app.route('/blog/<int:postID>')
def show_blog(postID):
   return 'Blog Number %d' % postID

@app.route('/rev/<float:revNo>')
def revision(revNo):
   return 'Revision Number %f' % revNo

@app.route("/preview")
def preview(filename):
    return 'File Uploaded'

@app.route("/predict")
def predict():
    return render_template('predict.html')

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
        #label_encoder_yy= pk.load(file)
        y_predict=trained_encoder.transform(df)
        y_pred=trained_model_for_prediction.predict(y_predict)
        #y_pred_oo_pred=label_encoder_yy.inverse_transform(y_pred)
        df['Days']=y_pred
        print(df)
        file.close()
        dataset1=df
        df.to_csv('./uploads_f/Predict1.csv')
        return render_template('upload0.html',tables=[dataset1.to_html(classes='dataset1')])
        #return 'Check Console'
    else:
        return 'Check with Expert'


@app.route('/function')
def get_ses():
    #Importing the dataset
    dataseta = pd.read_csv('./uploads_f/Train.tsv', sep='\t',dtype=object)
    dataset_test1 = pd.read_csv('./uploads_f/Test.tsv',sep='\t',dtype=object)
    dataseta.drop(dataseta[dataseta['Payer']=='*'].index,inplace=True)
    dataseta.drop(columns='PercentofPayAmount',inplace=True)
    dataset=dataseta[dataseta.duplicated(subset=["Payer","Debit_CreditInd.","DaystoPay"],keep='last')]
    dataset.info()
    dataset.describe()
    X1=dataset[['Payer','Debit_CreditInd.']]
    dataset_test=dataset_test1
    X1['Debit_CreditInd.']=dataset['Debit_CreditInd.'].apply(lambda x: 1 if x == 'S' else 0)
    dataset_test['Debit_CreditInd.']=dataset_test['Debit_CreditInd.'].apply(lambda x: 1 if x == 'S' else 0)
    y = dataset.iloc[:, -1].values
    file = open("models.pkl", "wb")
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'), [0])], remainder='passthrough')
    X = np.array(ct.fit_transform(X1))
    pk.dump(ct, file)
    from sklearn.ensemble import RandomForestClassifier
    classifier1 = RandomForestClassifier(n_estimators = 75, criterion = 'entropy', random_state = 0,bootstrap=True,max_leaf_nodes=None, max_samples=None,min_impurity_decrease=0.0, min_impurity_split=None,ccp_alpha=0.0, class_weight=None, max_depth=None, max_features='auto',min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0,n_jobs=None, oob_score=False, verbose=0,warm_start=False)
    classifier1.fit(X, y)
    pk.dump(classifier1, file)
    file.close()
    file = open("models.pkl", "rb")
    trained_encoder = pk.load(file)  #Pickle file first load the OneHotEncoder
    trained_model_for_prediction = pk.load(file)
    X_test_encoded=np.array(trained_encoder.transform(dataset_test))
    y_predict=trained_model_for_prediction.predict(X_test_encoded)
    dataset_test['DaystoPay']=y_predict
    print(dataset_test)
    file.close()
    y_pred=classifier1.predict(X)
    cm = confusion_matrix(y,y_pred)
    print( accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))
    dataset1=dataset_test
    dataset1.to_csv('./uploads_f/Predict.csv')
    return render_template('upload0.html',tables=[dataset1.to_html(classes='dataset1')])

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0',debug=True,port=5000)
