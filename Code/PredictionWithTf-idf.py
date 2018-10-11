#SYS 6018 Competition 3: Predicting Blogger Characteristics

#Importing the required packages
import numpy as np
import pandas as pd
import os
import nltk
import string
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from spellchecker import SpellChecker
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model import Lasso

#Downloading from nltk
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')

#Setting the working directory
os.getcwd()
os.chdir('E:/Fall Term/SYS 6018 Applied Data Mining/Competitions/sys6018-competition-blogger-characteristics/Data')

#Loading the dataset
traindata = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')

#Dropping the columns that are not unique in train and test data
train = traindata.drop(["post.id", "date","text"], axis=1)
test = testdata.drop(["post.id", "date","text"], axis=1)

#Creating the train dataframe with unique user.id observations
train.drop_duplicates(keep='first', inplace=True)
train.sort_values(by = 'user.id', inplace = True)
train.reset_index(drop = True, inplace = True)

#Creating the test dataframe with unique user.id observations
test.drop_duplicates(keep='first', inplace=True)
test.sort_values(by = 'user.id', inplace = True)
test.reset_index(drop = True, inplace = True)

#Grouping by user.id and joining text and storing in temp
temp = traindata.groupby(['user.id'])['text'].apply(','.join).reset_index()
#Merging temp and train to get the dataframe train
train = pd.merge(train, temp, on='user.id')

#Grouping by user.id and joining text and storing in tempt
tempt = testdata.groupby(['user.id'])['text'].apply(','.join).reset_index()
#Merging temp and train to get the dataframe train
test = pd.merge(test, tempt, on='user.id')

#Converting to categorical variables
cols = ['gender','topic', 'sign']
for col in train[cols]:
    train[col] = train[col].astype('category')
for col in test[cols]:
    test[col] = test[col].astype('category')

#Preparing the final train data by creating dummy variables for categorical variables
gender = pd.get_dummies(train['gender'])
topic = pd.get_dummies(train['topic'])
sign = pd.get_dummies(train['sign'])
train = pd.concat([train, gender, topic, sign], axis=1)

#Preparing the final test data
gender = pd.get_dummies(test['gender'])
topic = pd.get_dummies(test['topic'])
sign = pd.get_dummies(test['sign'])
test = pd.concat([test, gender, topic, sign], axis=1)

#Tokenizing the text in train and test data
wpt = nltk.WordPunctTokenizer()
train['tokenized'] = train['text'].apply(lambda x: wpt.tokenize(x))
test['tokenized'] = test['text'].apply(lambda x: wpt.tokenize(x))

#Getting sentence count, word count, character count, upper case count, punctuations count, urls count, hashtag count for train and test data
def countPunc(text):
    count = 0
    for i in text:
        if i in string.punctuation:
            count = count +1
    return count

def findurl(text):
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return len(url)

train['sentcount'] = train['text'].apply(lambda x: len(sent_tokenize(x)))
train['wordcount'] = train['text'].apply(lambda x: len(word_tokenize(x)))
train['charcount'] = train['text'].apply(lambda x: len(x))
train['uppercasecount'] = train['text'].apply(lambda x: sum(map(str.isupper, x)))
train['punccount'] = train['tokenized'].apply(lambda x: countPunc(x))
train['urlscount'] = train['text'].apply(lambda x: findurl(x))
train['hashtagcount'] = train['tokenized'].apply(lambda x: len([i for i in x if i.startswith('#')]))

test['sentcount'] = test['text'].apply(lambda x: len(sent_tokenize(x)))
test['wordcount'] = test['text'].apply(lambda x: len(word_tokenize(x)))
test['charcount'] = test['text'].apply(lambda x: len(x))
test['uppercasecount'] = test['text'].apply(lambda x: sum(map(str.isupper, x)))
test['punccount'] = test['tokenized'].apply(lambda x: countPunc(x))
test['urlscount'] = test['text'].apply(lambda x: findurl(x))
test['hashtagcount'] = test['tokenized'].apply(lambda x: len([i for i in x if i.startswith('#')]))

#Count of stop words
stop_words=set(stopwords.words("english"))

def countstopwords(tokenized):
    count = 0
    for w in tokenized:
        if w in stop_words:
            count = count +1
    return count

train['stopwordscount'] = train['tokenized'].apply(lambda x: countstopwords(x))
test['stopwordscount'] = test['tokenized'].apply(lambda x: countstopwords(x))

#Counting the misspelledwords
spell = SpellChecker()
train['misspelledwordscount'] = train['tokenized'].apply(lambda x: len([i for i in x if spell.unknown(i)]))
test['misspelledwordscount'] = test['tokenized'].apply(lambda x: len([i for i in x if spell.unknown(i)]))

#Pre processing the text data

#Keeping only words
train['tokenized'] = train['tokenized'].apply(lambda x: [i for i in x if i.isalpha()])
test['tokenized'] = test['tokenized'].apply(lambda x: [i for i in x if i.isalpha()])

#Converting everything to lower case
train['tokenized'] = train['tokenized'].apply(lambda x: [i.lower() for i in x])
test['tokenized'] = test['tokenized'].apply(lambda x: [i.lower() for i in x])

#Removing stop words
train['tokenized'] = train['tokenized'].apply(lambda x: [i for i in x if not i in stop_words])
test['tokenized'] = test['tokenized'].apply(lambda x: [i for i in x if not i in stop_words])

#Stemming the words

porter = PorterStemmer()

train['tokenized'] = train['tokenized'].apply(lambda x: [porter.stem(i) for i in x])
test['tokenized'] = test['tokenized'].apply(lambda x: [porter.stem(i) for i in x])

#Preprocessed text
train['ctext'] = train['tokenized'].apply(lambda x: " ".join(x))
test['ctext'] = test['tokenized'].apply(lambda x: " ".join(x))

#Feature Generation using TF-IDF
tf = TfidfVectorizer(max_features = 300)
train_tf = tf.fit_transform(train['ctext'])
test_tf = tf.fit_transform(test['ctext'])
#Converting to dataframe
train_tf = pd.DataFrame(train_tf.todense())
test_tf = pd.DataFrame(test_tf.todense())

#Merging TF-IDF features with the original data
train = pd.concat([train, train_tf], axis=1)
test = pd.concat([test, test_tf], axis=1)

#Dropping the text and tokenized columns from the train and test data
train = train.drop(["text", "tokenized", "ctext"], axis=1)
test = test.drop(["text", "tokenized", "ctext"], axis=1)

#Storing the user.id age of train separately
uid_tr = train['user.id']
age_tr = train['age']
uid_te = test['user.id']
#Dropping the user.id, age, gender, topic, sign (they have been added as dummy variables) from train data
train = train.drop(["user.id", "age", "gender","topic", "sign"], axis=1)
#Dropping the user.id from test data
test = test.drop(["user.id", "gender","topic", "sign"], axis=1)

#Standardizing the train and test data
std_train = pd.DataFrame(preprocessing.scale(train))
std_test = pd.DataFrame(preprocessing.scale(test))

#Log transformation of age
y = np.log(age_tr)

#Lasso Regression Model with 10 fold cross validation
reg = LassoCV(cv=10, random_state=0).fit(std_train, y.values.ravel())
reg.score(std_train, y)
#Predictions on the test data
predicted = np.exp(reg.predict(std_test))

#Preparing data for submission
submission = pd.DataFrame(columns = ['user.id','age'])
submission['user.id'] = uid_te
submission['age'] = predicted
submission.to_csv('submission.csv', index=False)
