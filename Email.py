import win32com.client 
import datetime as dt
import pandas as pd
import glob
import re

myFiles = glob.glob('*.xlsx')
print(myFiles)
for file in myFiles:
    print(file)
    df=pd.read_excel(file,header=0,dtype=object)

y = (dt.date.today() - dt.timedelta(days=30))
print(y)
y = y.strftime('%m/%d/%Y %H:%M %p')
print(y)

def isWordPresent(sentence, word):
    sentence = re.sub('[^a-zA-Z0-9]',' ',sentence)    
    s = sentence.split(" ")  
    for i in s: 
        if (i == word): 
            return True
    return False

outlook = win32com.client.Dispatch('Outlook.Application').GetNamespace('MAPI')
namespace = outlook.Session
recipient = namespace.CreateRecipient("deesaw@deloitte.com")
inbox = outlook.GetDefaultFolder(6)#(recipient, 6)
messages = inbox.Items
messages = messages.Restrict("[ReceivedTime] >= '" + y +"'")
email_subject = []

i=0
for x in messages:
    i=i+1
    sub = x.Subject
    if isWordPresent(sub,'Task'):
        email_subject.append(sub)
df1=pd.DataFrame()
for d in df['UID']:
    print(d)
    df1['UID']=d
    for e in email_subject:
        if isWordPresent(e,d):
            df1['Subject']=e


