# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:33:27 2018

@author: nicol
"""

import pandas as pd
import nltk
from nltk.corpus import brown 
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re
import string 
import statistics 
print('Imports done!')

# Read in data
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

#Seperate the age from the rest of the data because it's not a predictor 
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#Combine the train and test data 
train_y = train['age']
del train['age']
data = pd.concat([train,test], sort=False)
#There is no missing values in the dataset
data.isnull().values.any()
#The cutoff index between train and test 
cutoff = train.shape[0]
print('Datasets are loaded')


#make some general plots 
import matplotlib.pyplot as plt
import numpy as np  
temp = pd.concat([train, train_y], axis=1, join_axes=[train.index])
plt.style.use('seaborn')

factors=["gender","sign","topic","date"]
temp[factors] = temp[factors].astype('category')
temp.gender.value_counts()
temp.sign.value_counts()
temp.topic.value_counts()
temp.date.value_counts()


temp['logage'] = np.log(temp['age'])
temp.logage.hist()
train_y = temp['logage']
print('Finish data exploration')

#log version looks like more normalized 

#Let us get all the variables we need  

#Dummy variables for topic and gender 
data = pd.get_dummies(data,columns=["topic","gender"])

#Number of post per user 
data['numofpost'] = data.groupby('user.id')['post.id'].transform(lambda x: len(x))

#WokTok and Senttok with nltk packages
data['SentTok'] = data['text'].apply(lambda x: nltk.sent_tokenize(x))
data['WordTok'] = data['text'].apply(lambda x: nltk.word_tokenize(x))
NoWord = [',','(',')',':',';','.','%','{','}','[',']','!','?',"''","``",'$']
data['WordTok'] = data['WordTok'].apply(lambda x: [i for i in x if i not in NoWord])

#number of urls 
def findurl(text):
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return len(url)
data['NumUrls'] = data['text'].apply(lambda x: findurl(x))

#Percetnage of Puncuation in every sentence 
def countPunc(sentences):
    count = 0 
    for i in sentences:
        if i in string.punctuation:
            count = count +1 
    return count
data['PtPunctuation'] = data['SentTok'].apply(lambda x: sum([countPunc(i) for i in x])/len(x) if len(x)!=0 else 0)

#Number of hashtages in each post 
data['NumHashtag'] = data['WordTok'].apply(lambda x: len([i for i in x if i.startswith('#')]))

#Number of Uppercase words in each post 
data['NumUpper']= data['WordTok'].apply(lambda x: len([i for i in x if i.isupper()]))

#Clean the text to process more related variables 
data['text']=data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['text']=data['text'].str.replace('[^\w\s]','')

from nltk.corpus import stopwords 
stop = stopwords.words('english')
data['text']=data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#Check the number of misspelled words percentage in each post 
from nltk.corpus import webtext
dictionary1 =  set(brown.words())
dictionary2 = set(webtext.words())
dictionary = set.union(dictionary1, dictionary2)
data['NumMisspelling']=data['text'].apply(lambda x: len([x for x in x.split() if x not in dictionary])/len(x) if len(x)!=0 else 0)

#Number of characters and number of words 
data['NumCharacter'] = data['text'].apply(lambda x: sum([len(x) for x in x.split()]))
data['NumWord'] = data['text'].apply(lambda x: len([x for x in x.split()]))

import dateparser 
data['weekofday'] = data['date'].apply(lambda x: dateparser.parse(x ,date_formats = ['%d,%B,%Y'])).dt.weekday
#Get dummy vairables for the weekofday
data = pd.get_dummies(data,columns=["weekofday"]) 
print('Finish variable creation')



train_Y = pd.DataFrame(data=train_y)
train_Y.columns = ["age"] 
train_X = data[:cutoff]
test_X=data[cutoff:data.shape[0]] 



import matplotlib.pyplot as plt
plt.scatter(train_X[["NumCharacter"]], train_Y["age"])
plt.scatter(train_X[["NumWord"]], train_Y["age"])
#The two plots looks very similar, let us get rid of one of the variable Numword for both datasets 


train_X=train_X.drop(["sign","post.id","user.id","date","text","SentTok","WordTok","date","NumWord"], axis=1)
uid=test_X['user.id']
test_X=test_X.drop(["sign","post.id","user.id","date","text","SentTok","WordTok","date","NumWord"], axis=1)    




#Using standard scaler package to standerdize datasets  
train_X.numofpost.skew()
#2.4347606308529692
train_X.NumUrls.skew()
#36.95938171573091
train_X.PtPunctuation.skew()
#98.82756626715991
train_X.NumHashtag.skew()
#208.59626137386755
train_X.NumUpper.skew()
#155.82592403467459
train_X.NumMisspelling.skew()
#3.9281450587891262
train_X.NumCharacter.skew()
#47.07947226465115
train_X.NumWord.skew()
#48.16291726896521


#Normalize the variables who have high skewness (>5)
from sklearn.preprocessing import Normalizer
scaler = Normalizer()
scaler.fit(train_X[['NumUrls','PtPunctuation','NumHashtag','NumUpper','NumCharacter']])
train_X[['NumUrls','PtPunctuation','NumHashtag','NumUpper','NumCharacter']] = scaler.transform(train_X[['NumUrls','PtPunctuation','NumHashtag','NumUpper','NumCharacter']])
test_X[['NumUrls','PtPunctuation','NumHashtag','NumUpper','NumCharacter']] = scaler.transform(test_X[['NumUrls','PtPunctuation','NumHashtag','NumUpper','NumCharacter']])

test_X.to_csv('test_cleaned.csv')
train_X.to_csv('train_X_cleaned.csv')
train_Y.to_csv('train_Y_cleaned.csv')

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.2, random_state=0)
model = LinearRegression()
rfecv = RFECV(model, step=1, scoring='neg_mean_squared_error')
rfecv.fit(X_train, y_train.values.ravel())
# Recursive feature elimination
# Number of best features
rfecv.n_features_
# The number of best features is 56, which means based on recursvie cross validation result , we should use every feature. 
rfecv.support_
rfecv.ranking_
#Reduced X_test and X_train to the selected features 
rfecv.transform(X_train)
#Use the current model to predict X_test value 
ypred1 = rfecv.predict(X_test)
mean_squared_error(y_test.values,ypred1)
#  0.07404738031954539

#Maybe try another feature selection method 


from sklearn.feature_selection import SelectKBest , f_regression

uc = SelectKBest(f_regression, k =48)
f_regression1 = uc.fit(X_train, y_train.values.ravel())
features = f_regression1.transform(X_train)
features2= f_regression1.transform(X_test)
model2 = sm.OLS(y_train.values.ravel(),features).fit()
ypred2 = model2.predict(features2)
mean_squared_error(y_test.values,ypred2)
# 0.10057950433028898



#Try lasso crossvalidation 
from sklearn.linear_model import Lasso, LassoCV
reg = LassoCV(cv=5, random_state=0).fit(X_train, y_train.values.ravel())
reg.score(X_train, y_train) 
#0.2123729995581185
ypred3 = reg.predict(X_test)
mean_squared_error(y_test.values,ypred3)
# 0.07656058694273146

model3 = Lasso(normalize=True, max_iter=1e5)
model3 = model.fit(train_X, train_Y)
ypred4= model3.predict(X_test)
mean_squared_error(y_test.values,ypred4)
#0.07403005334623361



#Let us try lasso 
#Submission files should contain two columns: user.id (author's ID) and age (predicted age). 
#The user.id column should contain the user.id of all distinct authors in the test set.
test_X['age']=np.exp(model3.predict(test_X))
            
submission = pd.DataFrame(columns = ['user.id','age'])
test_X['user.id']=uid
test_X['avg']=test_X.groupby('user.id')['age'].transform(lambda x: statistics.mean(x))
test_X[['user.id','avg']]

test_X.drop_duplicates(subset='user.id', keep='first', inplace=True)
submission[['user.id','age']] = test_X[['user.id','avg']]
submission = submission.sort_values(by=['user.id'])
submission.to_csv('submission.csv',index=False)
print('Predictions out done!')

 

