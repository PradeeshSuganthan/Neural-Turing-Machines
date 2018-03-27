import numpy as np
import keras_preprocessing as kpp

from keras import callbacks, optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, GRU

#global constants
seq_size=30
sample_len=1000
char_level=True
verbose=1
batch_size=512
epochs=4
steps_per_epoch = None

#load data
x_plays, y_plays, plays_token = kpp.getPlaysAsListOfSequences(seq_size=seq_size, char_level=char_level, verbose=verbose)

vocabulary = len(plays_token.word_index) + 1

#create model
model = Sequential()

model.add(Embedding(vocabulary, 512))
model.add(GRU(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(GRU(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(GRU(512))
model.add(Dense(vocabulary, activation = 'softmax'))

print(model.summary())

adam = optimizers.Adam(lr=0.0001)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = adam, metrics = ['sparse_categorical_accuracy'])


path = 'saved_models/model_{epoch:02d}.h5'
checkpoint = callbacks.ModelCheckpoint(path, verbose=verbose)
reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=.5)

model.fit(x_plays,y_plays,epochs=epochs,verbose=verbose, batch_size = batch_size, steps_per_epoch=steps_per_epoch, validation_split=0.2, callbacks=[checkpoint, reduceLR])


print(kpp.genSequence(model, plays_token, seq_size=seq_size, char_level=char_level, verbose=verbose, sample_len=sample_len))