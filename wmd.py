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
# d = ["Obama speaks to the media in Illinois", "The President addresses the press in Chicago", 
# "It is holiday, enjoy my vacation in Chicago", "tomorrow is a holiday, i will take a rest", "I want to have a rest"]

vect = CountVectorizer(stop_words="english").fit([d1, d2])
# vect = CountVectorizer(stop_words="english").fit(d)
voc = [i for i,k in vect.vocabulary_.iteritems()]

# voc = [w for w in vect.get_feature_names()]

voc_vec = {i:wv[i] for i in voc}

from scipy.spatial.distance import cosine
v_1, v_2 = vect.transform([d1, d2])
# v = vect.transform(d)
# vv = np.zeros((5,14))
# for i in range(5):
# 	vv[i] = v[i].toarray().ravel()
v_1 = v_1.toarray().ravel()
v_2 = v_2.toarray().ravel()
print(v_1, v_2)
print("cosine(doc_1, doc_2) = {:.2f}".format(cosine(v_1, v_2)))

from sklearn.metrics import euclidean_distances
W_ = [voc_vec[w] for w in voc]
# D_ = euclidean_distances(W_)
D_ = np.ones((len(voc),len(voc)))
for i in range(len(voc)):
	for j in range(i):
		D_[i][j] = wv.similarity(voc[i], voc[j])
		D_[j][i] = D_[i][j]

print("d(addresses, chicago) = {:.2f}".format(D_[0, 1]))



import pickle
pickle.dump(voc_vec, open( "data/voc_vec.p", "wb" ))
voc_vec = pickle.load( open( "data/voc_vec.p", "rb" ) )

from pyemd import emd

# pyemd needs double precision input
v_1 = v_1.astype(np.double)
v_2 = v_2.astype(np.double)
v_1 /= v_1.sum()
v_2 /= v_2.sum()
# for i in range(5):
# 	vv[i] = vv[i].astype(np.double)
# 	vv[i] /= vv[i].sum()
D_ = D_.astype(np.double)
D_ /= D_.max()  # just for comparison purposes
print("d(doc_1, doc_2) = {:.2f}".format(emd(v_1, v_2, D_)))