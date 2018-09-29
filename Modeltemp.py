# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:15:55 2018
@author: Henry
"""
import pandas as pd
import nltk
#import statsmodels.api as sm
#import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn import linear_model
#from sklearn.metrics import mean_squared_error
print('Imports done!')

# Read in data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print('File Read Done!')

# Create Dummy Variables, fillna
cols = ['gender','topic','sign']
for col in train[cols]:
    train[col] = train[col].astype('category')
    train[col] = train[col].cat.codes
for col in test[cols]:
    test[col] = test[col].astype('category')
    test[col] = test[col].cat.codes
print('Cat Codes Done!')
#---------------------
# Training data cleaning and manipulation
train['SentTok'] = train['text'].apply(lambda x: nltk.sent_tokenize(x))
train['WordTok'] = train['text'].apply(lambda x: nltk.word_tokenize(x))
print('Progress is being made...')
NoWord = [',','(',')',':',';','.','%','{','}','[',']','!','?',"''","``",'$']
train['WordTok'] = train['WordTok'].apply(lambda x: [i for i in x if i not in NoWord])
train['NumSent'] = train['SentTok'].apply(lambda x: len(x))
train['NumWord'] = train['WordTok'].apply(lambda x: len(x))
train['SentLenAvg'] = train.apply(lambda x: x.NumWord / (x.NumSent + 1), axis=1)
train['WordLenAvg'] = train['text'].apply(lambda x: x.split())
train['WordLenAvg'] = train['WordLenAvg'].apply(lambda x: (sum(len(word) for word in x)/(len(x)+1)))
train.to_csv('trainPrep.csv',index=False)
print('NLKT on train done!')
#---------------------
"""
# Models
X_train, X_val, y_train, y_val = train_test_split(train.loc[:,train.columns != 'age'], train.age, test_size=0.2, random_state=0)
model = sm.OLS(y_train, X_train).fit()
ypred = model.predict(X_val)
mean_squared_error(y_val,ypred)
model.summary()

#---------------------
# Test data cleaning and manipulation
test['SentTok'] = test['text'].apply(lambda x: nltk.sent_tokenize(x))
test['WordTok'] = test['text'].apply(lambda x: nltk.word_tokenize(x))
NoWord = [',','(',')',':',';','.','%','{','}','[',']','!','?',"''","``",'$']
test['WordTok'] = test['WordTok'].apply(lambda x: [i for i in x if i not in NoWord])
test['NumSent'] = test['SentTok'].apply(lambda x: len(x))
test['NumWord'] = test['WordTok'].apply(lambda x: len(x))
test['SentLenAvg'] = test.apply(lambda x: x.NumWord / (x.NumSent + 1), axis=1)
test['WordLenAvg'] = test['text'].apply(lambda x: x.split())
test['WordLenAvg'] = test['WordLenAvg'].apply(lambda x: (sum(len(word) for word in x)/(len(x)+1)))
test.to_csv('testPrep.csv',index=False)
print('NLTK on test done!')
#--------------------

# Predictions for test data
test = test[['user.id','gender','topic']]
pred = model1.predict(test)

#output to df and csv
out = pd.DataFrame(columns = ['post.id','age'])
out['post.id'] = Ids
out['age'] = pred
out.to_csv('ModelOutput.csv',index=False)
print('Predictions out done!')
"""

