# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 13:05:03 2018

@author: Henry
"""
import pandas as pd
import nltk
import statsmodels.api as sm
import numpy as np
from textstat.textstat import textstat
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import random


train = pd.read_csv('trainPrep.csv')
test = pd.read_csv('testPrep.csv')
train = train.sample(n = 25000)

train['ReadingLvl'] = train['text'].apply(lambda x: textstat.flesch_reading_ease(x))
train.loc[train.ReadingLvl <= 0, 'ReadingLvl'] = 0
train.loc[train.ReadingLvl >= 100, 'ReadingLvl'] = 100

test['ReadingLvl'] = test['text'].apply(lambda x: textstat.flesch_reading_ease(x))
test.loc[test.ReadingLvl <= 0, 'ReadingLvl'] = 0
test.loc[test.ReadingLvl >= 100, 'ReadingLvl'] = 100


train.age = train.age.apply(np.log)

#--------------------------------------------
trainmod = train[['gender','age','topic','SentLenAvg','WordLenAvg','ReadingLvl']]

indexes = test[['post.id','user.id']]
testmod = test[['gender','topic','SentLenAvg','WordLenAvg','ReadingLvl']]

X_train, X_val, y_train, y_val = train_test_split(trainmod.loc[:,trainmod.columns != 'age'], trainmod.age, test_size=0.2, random_state=1)
model = sm.OLS(y_train, X_train).fit()
ypred = model.predict(X_val)
mean_squared_error(y_val,ypred)
model.summary()
#---------------------------

model1 = sm.OLS(trainmod.age, trainmod.loc[:,trainmod.columns != 'age']).fit()
preds = model1.predict(testmod)
preds = preds.apply(np.exp)

for n, i in enumerate(preds):
    if i < 13:
        preds[n] = random.randint(13,23)
    if i > 48:
        preds[n] = random.randint(38,48)

indexes['age'] = preds
grouped = indexes.groupby(['user.id']).age.mean().to_frame()
grouped['user.id'] = grouped.index
grouped = grouped[['user.id','age']]
grouped.to_csv('Model2.csv',index=False)


#-----------------------
histdata = [np.exp(x) for x in preds]
plt.xlim(0,50)
plt.hist(grouped.age[0:1000], bins = 'auto')
