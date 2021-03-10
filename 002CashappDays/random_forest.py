# -*- coding: utf-8 -*-
##Import Libraries
def model_train(filename=''):
    print('Getting Trained')
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    import pickle
    from sklearn.metrics import confusion_matrix, accuracy_score,average_precision_score,classification_report,f1_score
    from sklearn.compose import ColumnTransformer
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    import os

    #filepath = os.path.join('dataset','Train.tsv')
    filepath = os.path.join('dataset', filename)
    if os.path.isfile(filepath):
        print('udp', filepath)
        df_raw = pd.read_csv(filepath, sep='\t',dtype=object)
    else:
        print('abc')
        df_raw = pd.read_csv(r'dataset/Train.tsv', sep='\t',dtype=object)
    df = df_raw.copy()
    print(df.columns)
    df.columns=['Payer', 'Debit_CreditInd.', 'PercentofPayAmount', 'DaystoPay']
    df.drop(df[df['Payer']=='*'].index,inplace=True)
    df.drop(columns='PercentofPayAmount',inplace=True)
    Convert_dict = {'Payer': str,
                'Debit_CreditInd.': str,
                'DaystoPay': float
               }
    df = df.astype(Convert_dict)
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())
    dataseti=df.groupby(['Payer','Debit_CreditInd.'])['DaystoPay'].mean()
    datasetii = pd.DataFrame(dataseti).reset_index()
    inner_join = pd.merge(df,
                      datasetii,
                      on =['Payer','Debit_CreditInd.'],
                      how ='inner')
    inner_join['rank'] = inner_join.groupby(['Payer','Debit_CreditInd.'])['DaystoPay_x'].rank()
    dataset_ready=inner_join[['Payer','Debit_CreditInd.','rank','DaystoPay_x']]
    dataset=dataset_ready
    X1=dataset[['Payer','Debit_CreditInd.','rank']]

    #OnehotEncoding
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False,handle_unknown='ignore'), [0,1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X1))
    pickle.dump(ct,open('model_files/01ohe.sav', 'wb'))

    y = dataset.iloc[:, -1].values
    #Training Model
    classifier = RandomForestClassifier(n_estimators = 75, criterion = 'entropy', random_state = 0,bootstrap=True,
                                    ccp_alpha=0.0, class_weight=None, max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,n_jobs=None, oob_score=False, verbose=0,
                       warm_start=False)
    classifier.fit(X, y)
    pickle.dump(classifier,  open('model_files/02randomforest.sav', 'wb'))

    #Score
    y_pred = classifier.predict(X)
    y_prob = classifier.predict_proba(X)
    y_conf = [(np.max(pred_err)*100) for pred_err in y_prob]
    print(y_conf)
    cm = confusion_matrix(y, y_pred)
    print( accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))
    return(accuracy_score)



if __name__ == "__main__":
    model_train()
