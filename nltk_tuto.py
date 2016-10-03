#  coding: utf-8 
import os
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn import preprocessing

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

import logging


logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

# Test string #
string = 'Jadis, une nuit, je fus un papillon, voltigeant, content de son sort. Puis, je m’éveillai, étant Tchouang-tseu. Qui suis-je en réalité ? Un papillon qui rêve qu’il est Tchouang-tseu ou Tchouang qui s’imagine qu’il fut papillon ?'

# Have fun with tokenizers
tokenizer1 = nltk.data.load('tokenizers/punkt/french.pickle')
tokenizer2 = TreebankWordTokenizer()
french_stopwords = set(stopwords.words('french'))
stemmer = FrenchStemmer()

# See results
tokens1 = tokenizer1.tokenize(string)
tokens2 = tokenizer2.tokenize(string)
tokens3 = [token.encode('utf-8') for token in tokens2 if token.lower() not in french_stopwords]
tokens4 = [stemmer.stem(token.decode('utf-8')) for token in tokens3]


# Build class to add stem to pipleine

class StemmedCountVectorizer(CountVectorizer):

	def build_analyzer(self):
		analyzer = super(CountVectorizer, self).build_analyzer()
		return lambda doc:(stemmer.stem(w) for w in analyzer(doc))

analyzer = CountVectorizer().build_analyzer()
stem_vectorizer = StemmedCountVectorizer(stemmer)

def stemming(doc):
	return (stemmer.stem(w) for w in analyzer(doc))

# X = ['le chat est beau', 'le ciel est nuageux', 'les gens sont gentils', 'Paris est magique', 'Marseille est tragique', 'JCVD est fou']
# Y = [1,0,1,1,0,0]


#  Load data and shuffle it
data_set = pd.read_csv('data_set_4classes.csv')
data_set = data_set.iloc[np.random.permutation(len(data_set))]

X = data_set['doc']

# Labelencoder on the labels
labels = data_set['label']
le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)

# Define Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
					('tfidf', TfidfTransformer()),
					('clf', SVC())])

# parameters = { 'vect__analyzer': ['word', stemming]}
parameters = {}

gs_clf = grid_search.GridSearchCV(text_clf, parameters, n_jobs=-1, verbose = 1)
gs_clf.fit(X[::2], Y[::2]) 

print gs_clf.score(X[1::2],Y[1::2])
# vectorizer = StemmedCountVectorizer(stemmer)
# X_counts = vectorizer.fit_transform(X)
# tfidf_transformer = TfidfTransformer()
# X_tfidf = tfidf_transformer.fit_transform(X_counts)


program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)

