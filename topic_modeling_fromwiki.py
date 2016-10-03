# -*- coding: utf-8 -*-

import logging
import sys
import os
import cPickle
from gensim.utils import smart_open, simple_preprocess
from io import open
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import LdaModel
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary
from gensim.utils import ClippedCorpus
from gensim.corpora import MmCorpus
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

reload(sys)
sys.setdefaultencoding("utf-8")

#### Launch with python topic_modeling_fromwiki.py "the name of your list of documents"

start_time = time.time()
stops = set(stopwords.words('french'))

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in stops]

#################################
########## Load Data ############
#################################
inp = sys.argv[1]
data = cPickle.load( open( inp, "rb") )

### Create text_list ###
text_list = []
for text in data :
	tokens = tokenize(text)
	text_list.append(tokens)

dictionary = Dictionary(text_list)


### Create BOW corpus ###
corpus = [dictionary.doc2bow(text) for text in text_list]

print("--- Corpus made: %s minutes ---" % round(((time.time() - start_time)/60),2)) 



start_lda_time = time.time()

#################################
######### Train LDA  ############
#################################

lda_model = LdaMulticore(corpus, num_topics=4, id2word=dictionary, passes=150, workers = 3)
final_topics = lda_model.show_topics()

print("--- LDA trained : %s minutes ---" % round(((time.time() - start_lda_time)/60),2)) 


#################################
##### Display WordCloud #########
#################################
curr_topic = 0
wc = WordCloud(background_color="black", max_words=2000,max_font_size=40, width=120, height=120, random_state=42)
for line in final_topics:
    line = line[1]
    scores = [float(x.split("*")[0]) for x in line.split(" + ")]
    words = [x.split("*")[1] for x in line.split(" + ")]
    freqs = []
    for word, score in zip(words, scores):
        freqs.append((word, score))

   	elements = wc.fit_words(freqs)
    fig = plt.figure()
    plt.imshow(elements)
    plt.axis("off")
    fig.savefig('images/topic'+str(curr_topic))
    curr_topic += 1
    
plt.show()

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)


