import numpy as np
import keras_preprocessing as kpp

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, GRU, Dropout

#two words in, one word out sequence

x_plays, y_plays, plays_token = kpp.getPlaysAsListOfSequences(seq_size=50, char_level=True, verbose=1)

vocabulary = len(plays_token.word_index) + 1

#create model
model = Sequential()

model.add(Embedding(vocabulary, 500))
model.add(LSTM(500, return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(500))
model.add(Dense(vocabulary, activation = 'softmax'))

print(model.summary())

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['sparse_categorical_accuracy'])

model.fit(x_plays,y_plays,epochs=1,verbose = 1, batch_size = 128)

print(kpp.genSequence(model, plays_token, seq_size=50, char_level=True, verbose=1))