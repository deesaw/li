
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle as pk
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb

dataseta = pd.read_csv('Train.tsv',sep='\t',dtype=object)
dataset_test1 = pd.read_csv('Test.tsv',sep='\t',dtype=object)
dataseta.drop(dataseta[dataseta['Payer']=='*'].index,inplace=True)
dataseta.drop(columns='PercentofPayAmount',inplace=True)
#dataseta.info()
#dataseta.describe()
Convert_dict = {'Payer': str, 
                'Debit_CreditInd.': str,
                'DaystoPay': float
               } 
  
dataseta = dataseta.astype(Convert_dict) 
print(dataseta.dtypes) 
dataseta.head(5)
dataseti=dataseta.groupby(['Payer','Debit_CreditInd.'])['DaystoPay'].mean()
datasetii = pd.DataFrame(dataseti).reset_index()
#datasetii.head(20)
inner_join = pd.merge(dataseta,  
                      datasetii,  
                      on =['Payer','Debit_CreditInd.'],  
                      how ='inner') 
inner_join['rank'] = inner_join.groupby(['Payer','Debit_CreditInd.'])['DaystoPay_x'].rank()
dataset_ready=inner_join[['Payer','Debit_CreditInd.','rank','DaystoPay_x']]
dataset=dataset_ready
X1=dataset[['Payer','Debit_CreditInd.','rank']]
dataset_test=dataset_test1
X1['Debit_CreditInd.']=dataset['Debit_CreditInd.'].apply(lambda x: 1 if x == 'S' else 0)
dataset_test['Debit_CreditInd.']=dataset_test['Debit_CreditInd.'].apply(lambda x: 1 if x == 'S' else 0)
y = dataset.iloc[:, -1].values
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
file = open("models.pkl", "wb")
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X1))
pk.dump(ct, file)
from sklearn.neighbors import KNeighborsClassifier
classifier1 = KNeighborsClassifier(n_neighbors = 35,weights='uniform', metric = 'minkowski', p = 2)
classifier1.fit(X, y)
pk.dump(classifier1, file)
file.close()

dataset_test['rank']=1

file = open("models.pkl", "rb")
trained_encoder = pk.load(file)  #Pickle file first load the OneHotEncoder 
trained_model_for_prediction = pk.load(file)
X_test_encoded=np.array(trained_encoder.transform(dataset_test))
y_predict=trained_model_for_prediction.predict(X_test_encoded)
dataset_test['DaystoPay']=y_predict
print(dataset_test)
file.close()
y_pred=classifier1.predict(X)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import average_precision_score,classification_report
from sklearn.metrics import f1_score
#average_precision = average_precision_score(y_test_in, y_pred)
cm = confusion_matrix(y,y_pred)
print( accuracy_score(y,y_pred))
print(classification_report(y,y_pred))