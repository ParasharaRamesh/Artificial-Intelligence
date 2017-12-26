'''
in this example i am just training it for 1 epoch with 500 samples as i am doing it in virtual box i have memory problems
our final prediction is also not working for reasons i not sure of 
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.preprocessing.text import Tokenizer
import csv
import numpy
import random
from math import *
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support
from keras import backend as K
from numpy import array
from numpy import argmax
from numpy import array_equal
import gensim


#first extract the data
tweets=[]
with open('consolidated.csv') as f:
	csvreader=csv.reader(f,delimiter=',')
	for row in csvreader:
		if(row[4]=='1'):
			temp=[]
			temp.append(row[1])
			temp.append(row[2])
			tweets.append(temp)


print "csv parsing done!"
train=random.sample(tweets,500)#should be 10000
test=random.sample(tweets,500)
print "selected random sample of tweets"
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
print "testdata and traindata we have"
#for X train only

w2vxtrain=[]
for sentence in x_train:
	w2vxtrain.append(sentence.split())

model1 = gensim.models.Word2Vec(w2vxtrain, min_count=1,size=50)
#this is where word to vec happens!
padlen=50
newxtrain=[]
for sent in x_train:
	temp=[]
	sentlen=len(sent.split())
	for i in range(50-sentlen):
		temp.append(numpy.zeros(50))
	for word in sent.split():
		temp.append(model1[word])
	newxtrain.append(temp)
newxtrain=array(newxtrain)

print "new xtrain is done calculated"
print "starting to tokenize the y_train"
tokenizer = Tokenizer(num_words=3500)

tokenizer.fit_on_texts(y_train)
#print tokenizer[1]
word_index = tokenizer.word_index#mappinng of word to token no 
word_index["__unk__"]=3501
sequences = tokenizer.texts_to_sequences(y_train)
print "we have sequences now!"
data = pad_sequences(sequences, maxlen=50)
print " data is padded now"



newytrain=[]
for x in data:
	onehot=to_categorical(x,num_classes=3501)#num_classes is 3501
	newytrain.append(onehot)
newytrain=array(newytrain)

print "newytrain is calculated"



# configure problem
n_features = 50
n_timesteps_in = 50
n_timesteps_out =50

model = Sequential()

model.add(LSTM(150,input_shape=(n_timesteps_in, n_features)))
model.add(RepeatVector(n_timesteps_in))
model.add(LSTM(150, return_sequences=True))
model.add(TimeDistributed(Dense(3501, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print "model compiled!!"
print newxtrain.shape
print newytrain.shape
print model
model.fit(x=newxtrain,y=newytrain, batch_size=128,epochs=1)
#score = model.evaluate(newxtest, newytest, batch_size=128)
print "time to test with some bad tweet"
badtweet="hello how are u"
padlen=50
newxtest=[]
temp=[]
sentlen=len(badtweet.split())
for i in range(50-sentlen):
	temp.append(numpy.zeros(50))
for word in badtweet.split():
	temp.append(model1[word])
newxtest.append(temp)
newxtest=array(newxtest)
prediction=model.predict(newxtest)
print "prediction is"
print prediction
print "prediction shape is "
print prediction.shape
temp=[]
for x in prediction:
	for y in x:
		#tokenno=argmax(y)#not sure why it only prints tokennos of 0 but concept is understood
		print "tokenno",tokenno
		for k,v in word_index.iteritems():
			if(v==tokenno):
				temp.append(k)
				break
print "done"
print temp#this was supposed to give the good tweet but not sure why all words argmax is getting mapped to 0 but  concept is clear now!