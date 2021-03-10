# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 22:12:21 2020

@author: udprajapati
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:22:16 2020

@author: udprajapati
"""
##Import Libraries
def model_train(filename=''):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    
    from sklearn.ensemble import RandomForestClassifier
    
    from sklearn.metrics import classification_report
    import pickle
    import os
    import numpy as np
    ##Import Dataset
    filepath = os.path.join('dataset', filename)
    if os.path.isfile(filepath):  
        print('udp', filepath)
        df_raw = pd.read_csv(filepath)
    else:
        print('abc')
        df_raw = pd.read_csv(r'dataset/Dataset.csv')
        
    df = df_raw.copy()
    print(df.columns)
    df = df.drop('Account', axis=1)
    
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())
    
    ##Separate feature and target columns
    X = df.drop('Decision', axis=1)
    y = df['Decision'] 
    le_enc = LabelEncoder()
    y = le_enc.fit_transform(y)
    pickle.dump(le_enc, open('model_files/le_enc_tgt.sav', 'wb'))
    ##Split train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    
    ##Preprocessing
    #Categorical encoding.
    ohe_risk_cat = OneHotEncoder(drop='first')
    X_train_risk_enc = pd.DataFrame(ohe_risk_cat.fit_transform(X_train[['Risk Cat']]).toarray())
    X_train = X_train.reset_index().drop('index', axis=1)
    X_train = X_train.join(X_train_risk_enc)
    X_train = X_train.drop('Risk Cat', axis=1)
    
    X_test_risk_enc = pd.DataFrame(ohe_risk_cat.transform(X_test[['Risk Cat']]).toarray())
    X_test = X_test.reset_index().drop('index', axis=1)
    X_test = X_test.join(X_test_risk_enc)
    X_test = X_test.drop('Risk Cat', axis=1)
    
    pickle.dump(ohe_risk_cat,  open('model_files/ohe_risk_cat.sav', 'wb'))
    
    #Feature Scaling
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)
    pickle.dump(std_scaler,  open('model_files/std_scaler.sav', 'wb'))
    
    #Train model
    rf_classifier = RandomForestClassifier(n_estimators = 200, 
                                           criterion = 'entropy', 
                                           random_state = 0)
    rf_classifier.fit(X_train, y_train)
    pickle.dump(rf_classifier,  open('model_files/rf_classifier.sav', 'wb'))
    #Validate using test set
    y_pred = rf_classifier.predict(X_test)
    y_prob = rf_classifier.predict_proba(X_test)
    y_conf = [(np.max(pred_err)*100) for pred_err in y_prob]
    print(y_conf)
    
    #print the results
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    model_train()
