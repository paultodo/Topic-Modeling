#  coding: utf-8 
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import grid_search
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

import os.path as op
from glob import glob
import os
import sys
import logging

from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer, SnowballStemmer
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

#  Load data and shuffle it
df_data = pd.read_csv('SentimentAnalysisDataset.csv', error_bad_lines = False )[:500000]
data_set = df_data.iloc[np.random.permutation(len(df_data))]

X = df_data['SentimentText'].values
labels = df_data['Sentiment']
Y = labels
y = Y

french_stopwords = set(stopwords.words('french'))
english_stopwords = set(stopwords.words('english'))

stemmer = SnowballStemmer("english", ignore_stopwords = True)
analyzer = CountVectorizer().build_analyzer()

def stemmed_words(doc) :
    return (stemmer.stem(w) for w in analyzer(doc))

stem_vectorizer = CountVectorizer(analyzer = stemmed_words)

pipe1 = Pipeline([
    ('vect', stem_vectorizer),
    ('clf', MultinomialNB())])

pipe2 = Pipeline([
    ('vect', CountVectorizer(stop_words = french_stopwords)),
    ('clf', SVC())])

pipe3 = Pipeline([
    ('vect', stem_vectorizer),
    ('clf', xgb.XGBClassifier())])

mean_score_xgb = 0.
mean_score_mnb = 0.
mean_precision = 0.

n_folds = 5

for train_index, test_index in StratifiedKFold(y, n_folds=n_folds):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    pipe3.fit(x_train, y_train)
    score_xgb = pipe3.score(x_test, y_test)
    pipe1.fit(x_train, y_train) 
    score_Mnb = pipe1.score(x_test,y_test)
    print "score xgb cv", score_xgb
    print "score mnb cv", score_Mnb
    mean_score_xgb += score_xgb
    mean_score_mnb += score_Mnb
    # pred_mnb = pipe1.predict(x_test)
    # precision = accuracy_score(y_test, pred_mnb)
    # mean_precision += precision
    # print "accuracy mnb", precision
print "score XGB CV ", mean_score_xgb / 5.0
print "score MNB CV", mean_score_mnb / 5.0
# print "mean acc mnb", mean_precision / 5.0


