# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 17:49:54 2020

@author: deesaw
"""


import pandas as pd
import glob
import os

myFiles = glob.glob('*.txt')
print(myFiles)

for file in myFiles:
    separator=input("Choose 1 for tab separated file Choose 2 for comma separated text file")
    if separator=='1':
        df=pd.read_csv(file,sep='\t',dtype=object)
        print(file)
        df.to_excel(str(file.split('.')[0]+'.xlsx'),'Sheet1',engine='xlsxwriter',index=False)
    if separator=='2':
        df=pd.read_csv(file,sep=',',dtype=object)
        print(file)
        df.to_excel(str(file.split('.')[0]+'.xlsx'),'Sheet1',engine='xlsxwriter',index=False)
        
