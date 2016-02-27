import os

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split

from gensim.models.word2vec import Word2Vec
from sklearn.metrics import euclidean_distances

wv = Word2Vec.load_word2vec_format("../GoogleNews-vectors-negative300.bin.gz", binary = True)

d1 = "Obama speaks to the media in Illinois"
d2 = "The President addresses the press in Chicago"

vect = CountVectorizer(stop_words="english").fit([d1, d2])
voc = [i for i,k in vect.vocabulary_.iteritems()]

#r
# voc = [w for w in vect.get_feature_names()]

W_ = [wv[w] for w in voc]