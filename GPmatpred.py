# Random Forest Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

# Importing the dataset
dataset = pd.read_csv('Mat_Training.tsv', sep='\t',dtype=object)
print(dataset.isna().any(axis=0))
print(dataset.isnull().sum())
print(dataset['LIFNR'].unique())
print(dataset['MAT1'].unique())
print(dataset['MAT2'].unique())
print(dataset['MAT3'].unique())
print(dataset['SP_KUNNR'].unique())
print(dataset['MATNR'].unique())
#dataset.dropna()
print(dataset.info())
print(dataset.describe())


dataset1 = pd.read_csv('Mat_test.tsv', sep='\t')

dataset['istrainset'] = 1
dataset1['MATNR']=None
dataset1['istrainset'] = 0
dataset_p = pd.concat(objs=[dataset, dataset1], axis=0)



# Encoding categorical data

LIFNR_tr=pd.get_dummies(dataset_p[dataset_p['istrainset'] == 1]['LIFNR'],drop_first=True)
MAT1_tr=pd.get_dummies(dataset_p[dataset_p['istrainset'] == 1]['MAT1'],drop_first=True)
MAT2_tr=pd.get_dummies(dataset_p[dataset_p['istrainset'] == 1]['MAT2'],drop_first=True)
MAT3_tr=pd.get_dummies(dataset_p[dataset_p['istrainset'] == 1]['MAT3'],drop_first=True)
SP_KUNNR_tr=pd.get_dummies(dataset_p[dataset_p['istrainset'] == 1]['SP_KUNNR'],drop_first=True)


LIFNR_te=pd.get_dummies(dataset_p[dataset_p['istrainset'] == 0]['LIFNR'],drop_first=True)
MAT1_te=pd.get_dummies(dataset_p[dataset_p['istrainset'] == 0]['MAT1'],drop_first=True)
MAT2_te=pd.get_dummies(dataset_p[dataset_p['istrainset'] == 0]['MAT2'],drop_first=True)
MAT3_te=pd.get_dummies(dataset_p[dataset_p['istrainset'] == 0]['MAT3'],drop_first=True)
SP_KUNNR_te=pd.get_dummies(dataset_p[dataset_p['istrainset'] == 0]['SP_KUNNR'],drop_first=True)



X= pd.concat([LIFNR_tr,MAT1_tr,MAT2_tr,MAT3_tr,SP_KUNNR_tr],axis=1)
y = dataset['MATNR']


XX=pd.concat([LIFNR_te,MAT1_te,MAT2_te,MAT3_te,SP_KUNNR_te],axis=1)
yy=dataset1['MATNR']


label_encoder_y = preprocessing.LabelEncoder()
y1= label_encoder_y.fit_transform(y)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y1, test_size=0.10,random_state=1) 
                                                    
# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 25, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)

# Importing the dataset
y_pred1 = classifier.predict(X_test)
y_pred=label_encoder_y.inverse_transform(y_pred1)
y_test_in=label_encoder_y.inverse_transform(y_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test_in, y_pred)
print(cm)
print(accuracy_score(y_test_in, y_pred))

'''

y_predict1 = classifier.predict(XX)

y_predict_xls=label_encoder_y.inverse_transform(y_predict1)
print(y_predict_xls)
'''


