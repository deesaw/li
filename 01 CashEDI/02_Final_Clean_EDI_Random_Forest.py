# Random Forest Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle as pk
from sklearn.metrics import confusion_matrix, accuracy_score,average_precision_score,classification_report,f1_score

#Importing the dataset
dataseta = pd.read_csv('Train.tsv', sep='\t',dtype=object)
dataset=dataseta[dataseta.duplicated(subset=["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR"],keep='last')]
dataset= dataset.head(19000)
print(dataset.isna().any(axis=0))
print(dataset.isnull().sum())
#dataset.dropna()
print(dataset.info())
print(dataset.describe())

X1,y1=dataset[["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR"]],dataset["MATNR"]
label_encoder_y = preprocessing.LabelEncoder()
y= label_encoder_y.fit_transform(y1)
file = open("models.pkl", "wb")  
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'), [0,1,2,3,4])], remainder='passthrough')
X = np.array(ct.fit_transform(X1))
pk.dump(ct, file) #dumping Encoder model

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 75, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)
pk.dump(classifier, file)
pk.dump(label_encoder_y,file)
file.close()
cm = confusion_matrix(y, classifier.predict(X))
print( accuracy_score(y, classifier.predict(X)))
print(classification_report(y, classifier.predict(X)))

dataset1 = pd.read_csv('Test.tsv', sep='\t',dtype=object)
#dataset_test=ct.transform(dataset1)

#second method
file = open("models.pkl", "rb")
trained_encoder = pk.load(file)  #Pickle file first load the OneHotEncoder 
trained_model_for_prediction = pk.load(file)
label_encoder_yy= pk.load(file)
y_predict=trained_encoder.transform(dataset1)
y_pred=trained_model_for_prediction.predict(y_predict)
y_pred_oo_pred=label_encoder_yy.inverse_transform(y_pred)
dataset1['MATNR']=y_pred_oo_pred
print(dataset1)
file.close()
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
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

est = RandomForestClassifier(n_jobs=-1)
rf_p_dist={'max_depth':[15,20,40,60,70,80,None],
              'n_estimators':[70,80],
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               'min_samples_leaf':randint(1,60),
              }

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=9)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score

rf_parameters, rf_ht_score = hypertuning_rscv(est, rf_p_dist, 40, X, y)
print(rf_parameters)
print(rf_ht_score)

claasifier=RandomForestClassifier(bootstrap=False, criterion= 'gini', max_depth= 40 ,min_samples_leaf= 20, n_estimators= 80)

## Cross Validation good for selecting models
from sklearn.model_selection import cross_val_score

cross_val=cross_val_score(claasifier,X,y,cv=10,scoring='accuracy').mean()

