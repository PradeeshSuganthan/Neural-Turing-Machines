import numpy as np
import keras_preprocessing as kpp

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, GRU

#two words in, one word out sequence

x_plays, y_plays, plays_token = kpp.getPlaysAsListOfSequences(seq_size=10)

vocabulary = len(plays_token.word_index) + 1

#create model
model = Sequential()

model.add(Embedding(vocabulary, 500))
model.add(GRU(500, return_sequences=True))
model.add(LSTM(500))
model.add(Dense(vocabulary, activation = 'softmax'))

print(model.summary())

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['sparse_categorical_accuracy'])

model.fit(x_plays,y_plays,epochs=3,verbose = 1, batch_size = 200)

print(kpp.genSequence(model, plays_token, seq_size=10))