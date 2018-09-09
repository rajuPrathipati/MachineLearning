#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 22:19:33 2018

@author: nagaraju
"""
import pandas as pd
import numpy as np


import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('/home/nagaraju/Documents/ML/Adult_income/Adult_UCI',names=
                 ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','Income'])

#df = pd.read_csv('/home/nagaraju/Documents/ML/Adult_income/adult')


df.info()

df[df['workclass']=='?']

df.isnull().sum()





df.replace(to_replace=' ?',value=np.NaN,inplace=True)
df.dropna(inplace=True)

df['Income']=df['Income'].map({' <=50K':0,' >50K':1})


#df.dropna(subset=['workclass'],inplace=True)
#df.dropna(subset=['occupation'],inplace=True)
#df.dropna(subset=['native-country'],inplace=True)

df['workclass' ] =pd.Categorical(df['workclass']).codes
df['education' ] =pd.Categorical(df['education']).codes
df['marital-status' ] =pd.Categorical(df['marital-status']).codes
df['occupation' ] =pd.Categorical(df['occupation']).codes
df['relationship' ] =pd.Categorical(df['relationship']).codes
df['race' ] =pd.Categorical(df['race']).codes
df['sex' ] =pd.Categorical(df['sex']).codes
df['native-country' ] =pd.Categorical(df['native-country']).codes

df_male=df[df['sex']==1]
df_female=df[df['sex']==0]


sns.distplot(df['age'], bins=20, kde=False,rug=False)
plt.show()

fig = plt.figure(figsize=(8,5))
ax = sns.distplot(df['hours-per-week'])
ax.set_ylabel('Frequency')
plt.show()

df.info()

fig=plt.figure(figsize=(12,6))
ax=sns.countplot(x='hours-per-week',data=df,hue='Income')
plt.xticks(rotation=90)
plt.tight_layout()


df['age'].describe()

df.info()

df[df['native-country'].isnull()]

df['native-country'].value_counts()



print(df.Age.value_counts())
sns.boxplot(x='Age',y='Total_Purchase',data=df)


fig, ax = plt.subplots(figsize=(12,5))
ax = sns.countplot(x="sex", data=df,hue='Income')
plt.xticks(rotation=45)
plt.tight_layout()





x=df.iloc[:,:-1]#.drop(labels=['Administration','Marketing Spend','State_New York','State_Florida'],axis=1)
y=df.iloc[:,-1]


#x_opt = sm.add_constant(x)
model = sm.Logit(y,x)
results = model.fit()
print(results.summary())


df_male.drop(labels=['occupation'],axis=1,inplace=True)
df_male.drop(labels=['native-country'],axis=1,inplace=True)


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

lm=LogisticRegression()
lm.fit(x_train,y_train)

predict_y=lm.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predict_y))
print(confusion_matrix(y_test,predict_y))