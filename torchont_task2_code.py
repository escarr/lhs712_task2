# -*- coding: utf-8 -*-

import csv
import pandas as pd
import codecs
import sys
import re
import string
import emoji
import numpy as np
import itertools

from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

#Loading files
training_tsv = pd.read_csv("task2_en_training.tsv", encoding="utf-8", sep="\t")
validation_tsv = pd.read_csv("task2_en_validation.tsv", encoding="utf-8", sep="\t")

#View some tweets from training_tsv file
# print(training_tsv['tweet'].head(20))

#Removing punctuation

##https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression # got functions from link

def strip_links(text):
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text

def normalize_mentions(text):
    entity_prefixes = ['@']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
            else:
                words.append('@username')
    return ' '.join(words)

training_tsv['tweet_cleaned'] = training_tsv['tweet'].apply(lambda x: strip_links(x)) #remove links
training_tsv['tweet_cleaned'] = training_tsv['tweet_cleaned'].apply(lambda x: normalize_mentions(x)) # replace mentions with '@username'
training_tsv['tweet_cleaned'] = training_tsv['tweet_cleaned'].apply(lambda x: emoji.demojize(x, delimiters=('',''))) #replace emojis with words


tokenizer = RegexpTokenizer(r'\w+')
training_tsv['tweet_cleaned'] = training_tsv['tweet_cleaned'].apply(lambda x: ' '.join(tokenizer.tokenize(x))) #remove non alphanumeric characters
# training_tsv.to_csv("training_cleaned.csv", encoding="utf-8", index=False) #write cleaned data to file

###############################################################################

#Models


### Training/test split
# print(type(training_tsv['tweet']))
# print(type(training_tsv['tweet_cleaned']))
train_data, test_data = train_test_split(training_tsv, test_size=0.2)

label_tr = train_data['class']
label_te = test_data['class']
doc_ids_te = test_data['tweet_id']

###############################################################################
###Featurization

tfidf_vec = TfidfVectorizer()

###############################################################################
#Training on Training Data Set and Testing on Validation Data Set

print('\n'+'****Testing on Validation Data Set****'+'\n')

#SVM with Linear Kernel Original Training Data

label_tr = training_tsv['class']
label_val = validation_tsv['class']
doc_ids_tr = training_tsv['tweet_id']
doc_ids_val = validation_tsv['tweet_id']

x_tr = tfidf_vec.fit_transform(training_tsv['tweet'])
x_val = tfidf_vec.transform(validation_tsv['tweet'])

clf = svm.SVC(kernel='linear') #
clf.fit(x_tr,label_tr)
pred = clf.predict(x_val)

print ('\n'+'SVM (Linear Kernel; Trained with Original Data)')
print(sum(pred==label_val))
print(sum(pred!=label_val))
acc = accuracy_score(label_val, pred)
prec = precision_score(label_val, pred, zero_division=0)
rec = recall_score(label_val, pred)
f1 = f1_score(label_val, pred)

print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(rec))
print('F1 Score: {}'.format(f1))



#SVM with Linear Kernel Cleaned Training Data
x_tr_clean = tfidf_vec.fit_transform(training_tsv['tweet_cleaned'])
x_val = tfidf_vec.transform(validation_tsv['tweet'])

clf_clean = svm.SVC(kernel='linear')
clf_clean.fit(x_tr_clean,label_tr)
pred_clean = clf_clean.predict(x_val)

print('\n'+'SVM (Linear Kernel; Trained with Clean Data)')
print(sum(pred_clean==label_val))
print(sum(pred_clean!=label_val))
acc = accuracy_score(label_val, pred_clean)
prec = precision_score(label_val, pred_clean, zero_division=0)
rec = recall_score(label_val, pred_clean)
f1 = f1_score(label_val, pred_clean)

print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(rec))
print('F1 Score: {}'.format(f1))



#LinearSVC Clean Training
x_tr = tfidf_vec.fit_transform(training_tsv['tweet'])
x_val = tfidf_vec.transform(validation_tsv['tweet'])

clf = svm.LinearSVC(max_iter=2000) #
clf.fit(x_tr,label_tr)
pred = clf.predict(x_val)

print ('\n'+'Linear SVC (Trained with Original Data)')
print(sum(pred==label_val))
print(sum(pred!=label_val))
acc = accuracy_score(label_val, pred)
prec = precision_score(label_val, pred, zero_division=0)
rec = recall_score(label_val, pred)
f1 = f1_score(label_val, pred)

print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(rec))
print('F1 Score: {}'.format(f1))

#LinearSVC Clean Training
x_tr_clean = tfidf_vec.fit_transform(training_tsv['tweet_cleaned'])
x_val = tfidf_vec.transform(validation_tsv['tweet'])

clf_clean = svm.LinearSVC(max_iter=2000) #
clf_clean.fit(x_tr_clean,label_tr)
pred_clean = clf_clean.predict(x_val)

print('\n'+'Linear SVC (Trained with Clean Data)')
print(sum(pred_clean==label_val))
print(sum(pred_clean!=label_val))
acc = accuracy_score(label_val, pred_clean)
prec = precision_score(label_val, pred_clean, zero_division=0)
rec = recall_score(label_val, pred_clean)
f1 = f1_score(label_val, pred_clean)

print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(rec))
print('F1 Score: {}'.format(f1))



#Decision Trees Original Training
x_tr = tfidf_vec.fit_transform(training_tsv['tweet'])
x_val = tfidf_vec.transform(validation_tsv['tweet'])

DTclf = tree.DecisionTreeClassifier()#criterion='entropy', splitter ='random', min_samples_split=5)
DTclf = DTclf.fit(x_tr, label_tr)

pred=DTclf.predict(x_val)

print('\n' + 'Decision Trees (Trained with Original Data)')
print(sum(pred==label_val))
print(sum(pred!=label_val))
acc = accuracy_score(label_val, pred)
prec = precision_score(label_val, pred, zero_division=0)
rec = recall_score(label_val, pred)
f1 = f1_score(label_val, pred)

print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(rec))
print('F1 Score: {}'.format(f1))


#Decision Trees Trained with Cleaned Data
x_tr_clean = tfidf_vec.fit_transform(training_tsv['tweet_cleaned'])
x_val = tfidf_vec.transform(validation_tsv['tweet'])

DTclf = tree.DecisionTreeClassifier()#criterion='entropy', splitter='random', min_samples_leaf=10)
DTclf = DTclf.fit(x_tr_clean, label_tr)

pred_clean=DTclf.predict(x_val)

print('\n' + 'Decision Trees (Trained with Clean Data)')
print(sum(pred_clean==label_val))
print(sum(pred_clean!=label_val))

acc = accuracy_score(label_val, pred_clean)
prec = precision_score(label_val, pred_clean, zero_division=0)
rec = recall_score(label_val, pred_clean)
f1 = f1_score(label_val, pred_clean)

print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(rec))
print('F1 Score: {}'.format(f1))



#Random Forest Trained with Original Data

x_tr = tfidf_vec.fit_transform(training_tsv['tweet'])
x_val = tfidf_vec.transform(validation_tsv['tweet'])

RFclf = RandomForestClassifier()
RFclf = RFclf.fit(x_tr, label_tr)
pred= RFclf.predict(x_val)

print('\n' + 'Random Forest (Trained with Original Data)')
print(sum(pred==label_val))
print(sum(pred!=label_val))

acc = accuracy_score(label_val, pred)
prec = precision_score(label_val, pred, zero_division=0)
rec = recall_score(label_val, pred)
f1 = f1_score(label_val, pred)

print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(rec))
print('F1 Score: {}'.format(f1))



####Number of estimators is currently 500. Decrease for faster code
#Random Forest Trained with Cleaned Data
x_tr_clean = tfidf_vec.fit_transform(training_tsv['tweet_cleaned'])
x_val = tfidf_vec.transform(validation_tsv['tweet'])

RFclf = RandomForestClassifier()
RFclf = RFclf.fit(x_tr_clean, label_tr)
pred_clean= RFclf.predict(x_val)

print('\n' + 'Random Forest (Trained with Clean Data)')
print(sum(pred_clean==label_val))
print(sum(pred_clean!=label_val))

acc = accuracy_score(label_val, pred_clean)
prec = precision_score(label_val, pred_clean, zero_division=0)
rec = recall_score(label_val, pred_clean)
f1 = f1_score(label_val, pred_clean)

print('Accuracy: {}'.format(acc))
print('Precision: {}'.format(prec))
print('Recall: {}'.format(rec))
print('F1 Score: {}'.format(f1))
