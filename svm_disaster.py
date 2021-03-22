import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.svm import SVC
import re

df = pd.read_csv("train.csv", encoding = "utf-8")

text = df['text']   
target = df['target']

stop_words = list(stopwords.words('English'))
def clean(data, list_):
    for t in data:
        i = str(t)
        i = t.lower()
        i = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",i).split())
        i = re.sub(r'\W', ' ', str(i))
        i = re.sub(r'\<a href', ' ', i)
        i = re.sub(r'&amp;', '', i)
        i = re.sub(r'<br />', ' ', i)
        i = re.sub(r"^\s+|\s+$", "", i)
        i = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', i)
        i = re.sub(r'\'', ' ', i)
        i = re.sub(r'\s+[a-zA-Z]\s+', ' ', i)
        i = re.sub(r'\^[a-zA-Z]\s+', ' ', i)
        i = re.sub(r'\s+', ' ', i, flags=re.I)
        i = re.sub(r'^b\s+', '', i)
        i = re.sub(r"http\S+", "", i)
        i = i.split()
        i = [word for word in i if word not in stop_words]
        i = ' '.join(i)
        list_.append(i)
    
    return list_

X = []
X = clean(text, X)
y_train = np.array(target)

count = CountVectorizer()
X_train = count.fit_transform(X)

logr =  SVC(kernel = 'rbf')
history = logr.fit(X_train, y_train)

test_data = pd.read_csv("test.csv", encoding = "utf-8")["text"]
test = []
test = clean(test_data, test)

X_test = count.transform(test)

pred = logr.predict(X_test)
y_pred = np.array([1 if i > 0.5 else 0 for i in pred]).reshape(-1,1)
