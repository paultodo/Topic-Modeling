# -*- coding: utf-8 -*-

import logging
import sys
import os
import wikipedia
from wikipedia import page
from wikipedia import search, page
from wikipedia.exceptions import DisambiguationError
import cPickle
from gensim.parsing import PorterStemmer
import numpy as np
import time
import pandas as pd

reload(sys)
sys.setdefaultencoding("utf-8")

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

wikipedia.set_lang("fr")

global_stemmer = PorterStemmer()


### Corpus Mathématiques ###
title1 = "Apprentissage statistique"
title2 = "Géométrie"
title3 = "Division"
title4 = "Pythagore"
title5 = "multiplication"
title6 = "mathématiques"
title7 = "matrice aléatoire"
title8 = "gaussien"
title9 = "algèbre linéaire"
title10 = "probabilités"
### Corpus Littérature ### 
title11 = "coup du monde de football"
title12 = "tennis de table"
title13 = "natation"
title14 = "jeux olympiques"
title15 = "rugby à 15"
title16 = "boxe"
title17 = "football"
title18 = "aviron"
title19 = "équitation"
title20 = "escalade"

# themes = [title1, title2, title3, title4, title5,title6, title7, title8, title9, title10, title11, title12, title13,title14,title15,title16,title17,title18,title19,title20]
# numbers = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]


#### Easy test : creating corpus for LDA with 2 topics ####
title_test1 = "mathématiques"
title_test2 = "sport"
title_test3 = "littérature"
title_test4 = "cuisine"
themes = [title_test1,title_test2, title_test3, title_test4]
numbers = [250,250,250,250]

#######################################
######### Create a few methods ########
#######################################

def get_corpus(title,number) :
	# for a title and a number K, returns the title and the K first pages when looking searching for the title on wikipedia
	# results is a wikipage object
	wiki_pages = []
	titre = title
	titles = search(title, results = number)
	for i in range(number) :
		try :
			wiki_pages.append(wikipedia.page(titles[i]))
		except (DisambiguationError, IndexError) :
			print "ambiuigty or no results for title", title
	res = [titre,wiki_pages]
	return res

def get_text_from_corpus(corpus) : 
    # requires a corpus as input (with title and wikipages)
    # returns all the documents as a list of texts
    length = len(corpus[1])
    matrix = pd.DataFrame(index = range(length), columns = ['label','doc'])
    text =[]
    for i in range(length) :
        inter  = corpus[1][i].content.encode('utf8')
        text.append(inter)
        matrix['label'].iloc[i] = corpus[0]
        matrix['doc'].iloc[i] = inter

    return text, matrix


def get_global_corpus(themes,numbers,name):
	# given a list of themes and the number of pages you want for each, returns the complete list of texts
	# saves the results as "mon_corpus"
    start_time = time.time()
    total_text = []
    length = len(themes)
    matrix = pd.DataFrame(columns = ['label','doc'])
    for i in range(length) :
        corpus = get_corpus(themes[i], numbers[i])
        inter = get_text_from_corpus(corpus)
        matrix  = pd.concat((matrix, inter[1]), axis = 0, ignore_index = True)
        for j in range (numbers[i]) :
 			try :
				total_text.append(inter[0][j])

			except IndexError :
				pass
    cPickle.dump( total_text, open(name, "wb" ) )
    matrix.to_csv('data_set_4classes.csv' , index = False)
    print("--- Corpus made: %s minutes ---" % round(((time.time() - start_time)/60),2))	
    return total_text, matrix

class StemmingHelper(object):
    """
    Class to aid the stemming process - from word to stemmed form,
    and vice versa.
    The 'original' form of a stemmed word will be returned as the
    form in which its been used the most number of times in the text.
    """
 
    #This reverse lookup will remember the original forms of the stemmed
    #words
    word_lookup = {}
 
    @classmethod
    def stem(cls, word):
        """
        Stems a word and updates the reverse lookup.
        """
 
        #Stem the word
        stemmed = global_stemmer.stem(word)
 
        #Update the word lookup
        if stemmed not in cls.word_lookup:
            cls.word_lookup[stemmed] = {}
        cls.word_lookup[stemmed][word] = (
            cls.word_lookup[stemmed].get(word, 0) + 1)
 
        return stemmed
 
    @classmethod
    def original_form(cls, word):
        """
        Returns original form of a word given the stemmed version,
        as stored in the word lookup.
        """
 
        if word in cls.word_lookup:
            return max(cls.word_lookup[word].keys(),
                       key=lambda x: cls.word_lookup[word][x])
        else:
            return word

##################################
##### Try generating a corpus ####
##################################

text_total, matrice = get_global_corpus(themes,numbers,'mon_corpus_4classes')