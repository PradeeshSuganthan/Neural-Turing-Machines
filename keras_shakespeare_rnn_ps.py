import numpy as np
import keras_preprocessing as kpp

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

#two words in, one word out sequence

x_plays, y_plays, plays_token = kpp.getPlaysAsListOfSequences(seq_size=5)

vocabulary = len(plays_token.word_index) + 1

#create model
model = Sequential()

model.add(Embedding(vocabulary, 10))
model.add(LSTM(50))
model.add(Dense(vocabulary, activation = 'softmax'))

print(model.summary())

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_plays,y_plays,epochs=5,verbose = 1, batch_size = 100)

print(kpp.genSequence(model, plays_token))