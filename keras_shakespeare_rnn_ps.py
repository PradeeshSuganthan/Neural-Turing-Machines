import numpy as np
import random
import sys
import io
import keras_preprocessing as kpp

from keras import callbacks, optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, GRU

#global constants
seq_size=40
sample_len=1000
char_level=True
verbose=1
batch_size=1024
epochs=10
steps_per_epoch = None
step = 3

#load data
#x_plays, y_plays, plays_token = kpp.getPlaysAsListOfSequences(seq_size=seq_size, char_level=char_level, verbose=verbose)
files = kpp.import_files()
combined_plays = " ".join(files)
chars = sorted(list(set(combined_plays)))
char_indices = dict((c,i) for i,c in enumerate(chars))
indices_char = dict((i,c) for i,c in enumerate(chars))

vocabulary = len(chars)

print("Creating sequences...")
sentences = []
next_chars = []
for i in range(0,len(combined_plays)-seq_size,step):
    sentences.append(combined_plays[i:i+seq_size])
    next_chars.append(combined_plays[i+seq_size])

print("Creating training dataset...")
x = np.zeros((len(sentences),seq_size,vocabulary),dtype=np.bool)
y = np.zeros((len(sentences),vocabulary),dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i,t,char_indices[char]] = 1
    y[i,char_indices[next_chars[i]]] = 1

#create model
model = Sequential()

#model.add(Embedding(vocabulary, 32))
#model.add(LSTM(32, return_sequences=True))
#model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32, input_shape=(seq_size,vocabulary)))
model.add(Dense(vocabulary, activation = 'softmax'))

print(model.summary())

adam = optimizers.Adam()
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['categorical_accuracy'])

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
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
path = 'saved_models/model_{epoch:02d}.h5'
checkpoint = callbacks.ModelCheckpoint(path, verbose=verbose)
reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=.5)
print_sample = callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

#fit model
model.fit(x,y,epochs=epochs,verbose=verbose, batch_size = batch_size, steps_per_epoch=steps_per_epoch, 
          validation_split=0.2, callbacks=[print_sample, checkpoint, reduceLR])
