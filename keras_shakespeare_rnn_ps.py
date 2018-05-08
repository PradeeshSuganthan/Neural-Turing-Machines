import numpy as np
import random
import sys
import keras_rnn_preprocessing as kpp

from keras import callbacks, optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, GRU

#global constants
seq_size=40
sample_len=1000
char_level=True
verbose=1
batch_size=512
epochs=50
steps_per_epoch = None
step = 3
nietzsche = "./nietzsche/*.txt"
shakespeare = "./shakespeare/*.txt"

#load data
combined_plays, chars, vocabulary, char_indices, indices_char, x, y = kpp.get_data(seq_size, step, nietzsche)

#create model
model = Sequential()

#model.add(Embedding(vocabulary, 32))
model.add(LSTM(256, input_shape=(seq_size,vocabulary), return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256))
model.add(Dense(vocabulary, activation = 'softmax'))

print(model.summary())

adam = optimizers.Adam()
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['categorical_accuracy'])

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_sample(epoch, logs):
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(combined_plays) - seq_size - 1)
    generated = ''
    sentence = combined_plays[start_index: start_index + seq_size]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    print()
    print("-"*40 + 'Generated sequence' + '-'*40)
    sys.stdout.write(generated)
    for i in range(sample_len):
        x_pred = np.zeros((1, seq_size, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
    print("-"*40 + 'Terminate sequence' + '-'*40)
    print()

#create callbacks
path = 'saved_models/nieztsche_{epoch:02d}.h5'
checkpoint = callbacks.ModelCheckpoint(path, verbose=verbose)
reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=.5)
print_sample = callbacks.LambdaCallback(on_epoch_end=generate_sample)

#fit model
model.fit(x,y,epochs=epochs,verbose=verbose, batch_size = batch_size, steps_per_epoch=steps_per_epoch, 
          validation_split=0.2, callbacks=[print_sample, checkpoint, reduceLR])