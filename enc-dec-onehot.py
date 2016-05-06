from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from __future__ import print_function
from keras.models import Sequential
from keras.engine.training import slice_X
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector
from keras.layers import recurrent
import numpy as np

class CharacterTable(object):
    '''
        Given a set of characters:
        + Encode them to a one hot integer representation
        + Decode the one hot integer representation to their character output
        + Decode a vector of probabilties to their character output
        '''
    def __init__(self, vocab, maxlen):
        self.vocab = vocab
        self.char_indices = dict((c, i) for i, c in enumerate(self.vocab))
        self.indices_char = dict((i, c) for i, c in enumerate(self.vocab))
        self.maxlen = maxlen
    
    def encode(self, C, maxlen=None):
        maxlen = maxlen if maxlen else self.maxlen
        X = np.zeros((maxlen, len(self.vocab)))
        for i, c in enumerate(C):
            try:
                X[i, self.char_indices[c]] = 1
            except KeyError:
                X[i, self.char_indices[' ']] = 1
        return X
    
    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ' '.join(self.indices_char[x] for x in X)


import re
token_pattern=r"(?u)\b\w\w+\b"
def build_tokenizer():
    """Return a function that splits a string into a sequence of tokens"""
    pattern = re.compile(token_pattern)
    return lambda doc: pattern.findall(doc)


def readData(src):
    b1 = []
    b2 = []
    with open(src) as p:
        for i, line in enumerate(p):
            s = line.split('\t')
            if len(s) == 2:
                b1.append(s[0])
                b2.append(s[1][:-1]) #remove \n
                lines = i + 1
    return b1, b2, lines


def readGs(src):
    b = []
    with open(src) as p:
        for i, line in enumerate(p):
            b.append(round(float(line),0))
            lines = i + 1
    return b, lines

msr = './dataset/STS2012-train/STS.input.MSRpar.txt'
msrvid = './dataset/STS2012-train/STS.input.MSRvid.txt'
smt = './dataset/STS2012-train/STS.input.SMTeuroparl.txt'
b1_12_1, b2_12_1, l_12_1 = readData(msr)
print l_12_1
b1_12_2, b2_12_2, l_12_2 = readData(msrvid)
print l_12_2
b1_12_3, b2_12_3, l_12_3 = readData(smt)
print l_12_3
lines_12 = l_12_1 + l_12_2 + l_12_3
b1_12_train = b1_12_1 + b1_12_2 + b1_12_3
b2_12_train = b2_12_1 + b2_12_2 + b2_12_3
print lines_12


msr_gs = './dataset/STS2012-train/STS.gs.MSRpar.txt'
msr_gs_vid = './dataset/STS2012-train/STS.gs.MSRvid.txt'
smt_gs = './dataset/STS2012-train/STS.gs.SMTeuroparl.txt'
b_12_train = readGs(msr_gs)[0]
b_12_train = b_12_train + readGs(msr_gs_vid)[0]
b_12_train = b_12_train + readGs(smt_gs)[0]
print len(b_12_train) == len(b1_12_train) == len(b2_12_train)

msr_test = './dataset/STS2012-test/STS.input.MSRpar.txt'
vid_test = './dataset/STS2012-test/STS.input.MSRvid.txt'
smt_test = './dataset/STS2012-test/STS.input.SMTeuroparl.txt'
surprise_test = './dataset/STS2012-test/STS.input.surprise.OnWN.txt'
surprise2_test = './dataset/STS2012-test/STS.input.surprise.SMTnews.txt'
b1_12_1t, b2_12_1t, l_12_1t = readData(msr_test)
print l_12_1t
b1_12_2t, b2_12_2t, l_12_2t = readData(vid_test)
print l_12_2t
b1_12_3t, b2_12_3t, l_12_3t = readData(smt_test)
print l_12_3t
b1_12_4t, b2_12_4t, l_12_4t = readData(surprise_test)
print l_12_4t
b1_12_5t, b2_12_5t, l_12_5t = readData(surprise2_test)
print l_12_5t
lines = l_12_1t + l_12_2t + l_12_3t + l_12_4t + l_12_5t
b1_12_test = b1_12_1t + b1_12_2t + b1_12_3t + b1_12_4t + b1_12_5t
b2_12_test = b2_12_1t + b2_12_2t + b2_12_3t + b2_12_4t + b2_12_5t
print lines


msr_test_gs = './dataset/STS2012-test/STS.gs.MSRpar.txt'
vid_test_gs = './dataset/STS2012-test/STS.gs.MSRvid.txt'
smt_test_gs = './dataset/STS2012-test/STS.gs.SMTeuroparl.txt'
surprise_test_gs = './dataset/STS2012-test/STS.gs.surprise.OnWN.txt'
surprise2_test_gs = './dataset/STS2012-test/STS.gs.surprise.SMTnews.txt'
b_12_test = readGs(msr_test_gs)[0]
b_12_test = b_12_test + readGs(vid_test_gs)[0]
b_12_test = b_12_test + readGs(smt_test_gs)[0]
b_12_test = b_12_test + readGs(surprise_test_gs)[0]
b_12_test = b_12_test + readGs(surprise2_test_gs)[0]
print len(b_12_test) == len(b1_12_test) == len(b2_12_test)

t14_f = './dataset/STS2014-test/STS.input.deft-forum.txt'
t14_n = './dataset/STS2014-test/STS.input.deft-news.txt'
t14_h = './dataset/STS2014-test/STS.input.headlines.txt'
t14_i = './dataset/STS2014-test/STS.input.images.txt'
t14_o = './dataset/STS2014-test/STS.input.OnWN.txt'
t14_t = './dataset/STS2014-test/STS.input.tweet-news.txt'
b1_14_1t, b2_14_1t, l_14_1t = readData(t14_f)
print l_14_1t
b1_14_2t, b2_14_2t, l_14_2t = readData(t14_n)
print l_14_2t
b1_14_3t, b2_14_3t, l_14_3t = readData(t14_h)
print l_14_3t
b1_14_4t, b2_14_4t, l_14_4t = readData(t14_i)
print l_14_4t
b1_14_5t, b2_14_5t, l_14_5t = readData(t14_o)
print l_14_5t
b1_14_6t, b2_14_6t, l_14_6t = readData(t14_t)
print l_14_6t
b1_14_test = b1_14_1t + b1_14_2t + b1_14_3t + b1_14_4t + b1_14_5t + b1_14_6t
b2_14_test = b2_14_1t + b2_14_2t + b2_14_3t + b2_14_4t + b2_14_5t + b2_14_6t
lines = l_14_1t + l_14_2t + l_14_3t + l_14_4t + l_14_5t + l_14_6t
print lines

t14_f_gs = './dataset/STS2014-test/STS.gs.deft-forum.txt'
t14_n_gs = './dataset/STS2014-test/STS.gs.deft-news.txt'
t14_h_gs = './dataset/STS2014-test/STS.gs.headlines.txt'
t14_i_gs = './dataset/STS2014-test/STS.gs.images.txt'
t14_o_gs = './dataset/STS2014-test/STS.gs.OnWN.txt'
t14_t_gs = './dataset/STS2014-test/STS.gs.tweet-news.txt'
b_14_test = readGs(t14_f_gs)[0]
b_14_test = b_14_test + readGs(t14_n_gs)[0]
b_14_test = b_14_test + readGs(t14_h_gs)[0]
b_14_test = b_14_test + readGs(t14_i_gs)[0]
b_14_test = b_14_test + readGs(t14_o_gs)[0]
b_14_test = b_14_test + readGs(t14_t_gs)[0]
print len(b_14_test) == len(b1_14_test) == len(b2_14_test)

b1 = b1_12_train + b1_12_test + b1_14_test
b2 = b2_12_train + b2_12_test + b2_14_test
y_train = b_12_train + b_12_test + b_14_test
print len(b1) == len(b2) == len(y_train)
len(b1)

vectorizer = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(b1 + b2)
vectors.shape
vocab = vectorizer.get_feature_names()
len(vocab)


MAXLEN = 20
vocab.append(' ') #add empty word for padding
X = np.zeros((len(b1), MAXLEN, len(vocab)), dtype=np.bool)
y = np.zeros((len(b1), MAXLEN, len(vocab)), dtype=np.bool)

b1 = [x.lower() for x in b1]
b2 = [x.lower() for x in b2]
tokenize = build_tokenizer()
b1_tokens = [tokenize(x)[:MAXLEN] for x in b1]
b2_tokens = [tokenize(x)[:MAXLEN] for x in b2]
#padding
b1_tokens = [s + [' '] * (MAXLEN - len(s)) for s in b1_tokens]
b2_tokens = [s + [' '] * (MAXLEN - len(s)) for s in b2_tokens]
#Reverse
b1_tokens = [s[::-1] for s in b1_tokens]
b2_tokens = [s[::-1] for s in b2_tokens]

ctable = CharacterTable(vocab, MAXLEN)
for i, sentence in enumerate(b1_tokens):
    X[i] = ctable.encode(sentence, maxlen=MAXLEN)

for i, sentence in enumerate(b2_tokens):
    y[i] = ctable.encode(sentence, maxlen=MAXLEN)

HIDDEN_SIZE = 256
BATCH_SIZE = 128
LAYERS = 4

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(LSTM(HIDDEN_SIZE, dropout_W=0.5, dropout_U=0.1, input_shape=(MAXLEN, len(vocab))))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(MAXLEN))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(LSTM(HIDDEN_SIZE, dropout_W=0.5, dropout_U=0.1, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributedDense(len(vocab)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adagrad')

model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=3,
          show_accuracy=True,validation_split = 0.1, shuffle=True)

ctable.decode(y[0])
res = model.predict_classes(X[:10])
res_sentences = []
for r in res:
    sent = []
    for i in range(MAXLEN):
        sent.append(ctable.indices_char[r[i]])
    res_sentences.append(sent)

res_sentences


