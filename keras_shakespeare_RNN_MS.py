import sys
from matplotlib import pyplot
sys.path.insert(0, './')
from keras.preprocessing import text
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LSTM, Embedding
import keras_preprocessing as kpp

seq_length = 50


x_plays, y_plays, plays_token = kpp.getPlaysAsListOfSequences(seq_size=seq_length)
print('got data')
vocab_size = len(plays_token.word_index) +1


model = Sequential()
model.add(Embedding(vocab_size, 50, mask_zero=True, input_length=seq_length))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy',  optimizer='adam', metrics=['sparse_categorical_accuracy'])


history = model.fit(x_plays[:10000, :], y_plays[:10000], verbose=1, epochs=1, batch_size=2048)


print(kpp.genSequence(model, plays_token, seq_length))

pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()