import sys
sys.path.insert(0, './')
from keras.preprocessing import text
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, LSTM
import keras_preprocessing as kpp

x_plays, y_plays, plays_token = kpp.getPlays()
model = Sequential()
model.add(LSTM(50, input_length=1))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(x_onehot, y_onehot, epochs=50, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()