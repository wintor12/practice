from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils


categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
# newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)
# newsgroups_test = fetch_20newsgroups(subset='test',categories = categories)
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
vectorizer = TfidfVectorizer(max_df = 0.95, min_df = 5, stop_words = 'english')
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors_test = vectorizer.transform(newsgroups_test.data)
print vectors.shape
print vectors.shape[1]
print newsgroups_train.target.shape

y_train, y_test = [np_utils.to_categorical(x) for x in (newsgroups_train.target, newsgroups_test.target)]
X_train = vectors.todense() #can not input sparse matrix
X_test = vectors_test.todense()

model = Sequential()
model.add(Dense(30, input_dim=vectors.shape[1], init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(30, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(20, init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd)


model.fit(X_train, y_train,
          nb_epoch=20,
          batch_size=16,
          show_accuracy=True)
score = model.evaluate(X_test, y_test, batch_size=16, show_accuracy=True)