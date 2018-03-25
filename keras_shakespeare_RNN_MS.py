import sys
from matplotlib import pyplot
sys.path.insert(0, './')
from keras.preprocessing import text
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LSTM, Embedding, GRU
import keras_preprocessing as kpp


seq_size = 50
sample_length =500
char_level=True
batch_size = 256
epochs = 1
verbose = 1

x_plays, y_plays, plays_token = kpp.getPlaysAsListOfSequences(char_level=char_level,seq_size=seq_size, verbose=verbose)

vocab_size = len(plays_token.word_index) +1

if verbose:
    print('Building Model')
model = Sequential()
model.add(Embedding(vocab_size, 500, mask_zero=True, input_length=seq_size))
model.add(LSTM(500))
model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',  optimizer='adam', metrics=['sparse_categorical_accuracy'])


history = model.fit(x_plays[:1000,:], y_plays[:1000], verbose=1, epochs=epochs, batch_size=batch_size)


print(kpp.genSequence(model, plays_token, sample_length, seq_size, char_level, verbose=verbose))
