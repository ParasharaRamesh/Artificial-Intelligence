from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
import json
import numpy
import random
from math import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support
from keras import backend as K
#first extract the data
data = {}
pos=[]
neg=[]
posneg=[]
with open('pos_amazon_cell_phone.json') as f:
	data=json.load(f)
	for x in data["root"]:
		temp=[]
		temp.append(x["summary"])
		temp.append(x["rating"])
		pos.append(temp)
		posneg.append(temp)
		
with open('neg_amazon_cell_phone.json') as f:
	data=json.load(f)
	for x in data["root"]:
		temp=[]
		temp.append(x["summary"])
		temp.append(x["rating"])
		neg.append(temp)
		posneg.append(temp)
		
print "json parsing done!"

train=random.sample(posneg,3000)
test=random.sample(posneg,250)
X_train=[]
Y_train=[]
X_test=[]
Y_test=[]
for x in train:
	X_train.append(x[0])
	Y_train.append(x[1])
for x in test:
	X_test.append(x[0])
	Y_test.append(x[1])
x_train=numpy.array(X_train)
y_train=numpy.array(Y_train)
x_test=numpy.array(X_test)
y_test=numpy.array(Y_test)



embeddings_index = {}
f = open("/home/parashara/glove.6B.50d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = numpy.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
#print embeddings_index['hello']
print('Found %s word vectors.' % len(embeddings_index))

tokenizer = Tokenizer(num_words=20000)

x_train = [s.encode('ascii') for s in x_train]
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
print x_train[:3]
print "sequences are"
print sequences[0:4]

x_test = [s.encode('ascii') for s in x_test]
testsequences = tokenizer.texts_to_sequences(x_test)
testdata = pad_sequences(testsequences, maxlen=100)
print "testsequences are"
print testsequences[0:4]
print "testdata is"
print testdata[0:4]

testlabels = to_categorical(y_test)
print testlabels[:3]
print len(testlabels)
print testdata[:3]
print len(testdata)
#print sequences[:20]
word_index = tokenizer.word_index
#print word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=100)
print data[:2]

labels = to_categorical(y_train)
print labels[:2]
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# prepare embedding matrix
num_words = len(word_index)+1 
embedding_matrix = numpy.zeros((num_words,50))
print len(embedding_matrix)
for word, i in word_index.items():
	
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		#print i
        # words not found in embedding index will be all-zeros.
		embedding_matrix[i] = embedding_vector

#print embedding_matrix[:3]


num_validation_samples = int(0.2 * data.shape[0])



x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

a=tokenizer.texts_to_sequences(["this is brilliant excellent and great good"])

print a
b=pad_sequences(a,maxlen=100)
print b



model = Sequential()
model.add(Embedding(num_words,50,weights=[embedding_matrix],input_length=100,trainable=False))
model.add(LSTM(32))
model.add(Dropout(0.8))
model.add(Dense(6, activation='sigmoid'))

def f_score_obj(y_true, y_pred):
    y_true = K.eval(y_true)
    y_pred = K.eval(y_pred)
    precision, recall, f_score, support = precision_recall_fscore_support(y_true, y_pred)
    return K.variable(1.-f_score[1])


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall




model.compile(loss='binary_crossentropy',optimizer='adamax',metrics=['accuracy',recall,precision])

model.fit(data,labels, batch_size=128,epochs=8,validation_data=(x_val, y_val))
score = model.evaluate(testdata, testlabels, batch_size=128)
print "##################"
print score
print "#################"


#print model.predict(b)

