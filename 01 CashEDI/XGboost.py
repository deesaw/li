# XGBoost

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

# Importing the dataset
dataset = pd.read_csv('Mat_Training.tsv', sep='\t',dtype=object)
dataset=dataset[dataset.duplicated(subset=["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR"],keep='last')]

X_raw = dataset[['LIFNR', 'MAT1', 'MAT2','MAT3','SP_KUNNR']]
y_raw = dataset["MATNR"]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0,1,2,3,4])], remainder='passthrough')
X = ct.fit_transform(X_raw)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(y_raw)


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=75,criterion='entropy',random_state=0)
classifier.fit(X,y)

from sklearn.metrics import accuracy_score,classification_report,f1_score,precision_score,recall_score,confusion_matrix

#X=ct.inverse_transform(X)
y=le.inverse_transform(y)
y_predict=classifier.predict(X)
y_predict=le.inverse_transform(y_predict)
print("Accuracy: {:.2f} %".format(accuracy_score(y,y_predict)*100))
print(classification_report(y,y_predict))
#print("precision_score: {:.5f} %".format(precision_score(y,y_predict)))
#print("recall_score: {:.5f} %".format(recall_score(y,y_predict)))
#print("f1_score: {:.5f} %".format(f1_score(y,y_predict)))

dataset_test_input_data = pd.read_csv('Mat_test.tsv', sep='\t')
X_test_input_data=dataset_test_input_data[['LIFNR', 'MAT1', 'MAT2','MAT3','SP_KUNNR']]
#y_test_input_data=dataset_test_input_data['MATNR']

X_test_input_data_ohed=ct.transform(X_test_input_data)
y_predict_test_input_data=classifier.predict(X_test_input_data_ohed)

y_predict_test_input_data_compare=le.inverse_transform(y_predict_test_input_data)

#takes time
#cm=confusion_matrix(y,y_predict)
#import seaborn as sns
#sns.heatmap(cm, annot=True)
'''
# Applying Grid Search to find the best model and the best parameters
#Best Parameters: {'criterion': 'entropy', 'max_depth': 60, 'n_estimators': 60}
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[75],'criterion' :['entropy'],'max_depth' :[55]},
              {'n_estimators':[60],'criterion' : ['entropy'],'max_depth' :[50,60]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X, y)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
'''

'''
X=np.array(X).astype('float32')
y=np.array(y).astype('float32')
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X, y)
y_train=classifier.predict(X)



from sklearn.metrics import confusion_matrix, accuracy_score,average_precision_score,f1_score

print(accuracy(y,le.inverse_transform(y_train)))

dataset_test = pd.read_csv('Mat_test.tsv', sep='\t')
X_test=ColumnTransformer(dataset_test)
y_test=classifier.predict(X_test)
y_test=le.inverse_transform(y_test)




# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
'''