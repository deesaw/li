# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 17:50:25 2020

@author: deesaw
"""

import pandas as pd
import glob
import os
                
#os.chdir(r'D:\Users\DESAW\Documents\My Received Files\LOAD')
myFiles = glob.glob('*.txt')
print(myFiles)

for file in myFiles:
    df=pd.read_csv(file,sep='\t',dtype=object)
    print(file)
    column_name=input("Enter column name to be used for split")
    Warehouse=df[column_name].unique()
    print(Warehouse)
    for wh in Warehouse: 
        df1=df[df[column_name]==wh]
        df1.to_excel(str(file.split('.')[0]+wh+'.xlsx'),'Sheet1',engine='xlsxwriter',index=False)
