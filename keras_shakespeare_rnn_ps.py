import numpy as np
import keras_preprocessing as kpp

from keras import callbacks, optimizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, GRU

#global constants
seq_size=5
sample_len=500
char_level=True
verbose=1
batch_size=1024
epochs=2
steps_per_epoch = None

#load data
x_plays, y_plays, plays_token = kpp.getPlaysAsListOfSequences(seq_size=seq_size, char_level=char_level, verbose=verbose)

vocabulary = len(plays_token.word_index) + 1

#create model
model = Sequential()

model.add(Embedding(vocabulary, 32))
#model.add(LSTM(32, return_sequences=True))
#model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(vocabulary, activation = 'softmax'))

print(model.summary())

adam = optimizers.Adam()
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = adam, metrics = ['sparse_categorical_accuracy'])


def on_epoch_end(epoch, logs):
    print("This is a test")


def sequence_to_text(seq, token, char_level=False):
    wordmap = {v: k for k,v in token.word_index.items()}
    script = ""
    for word in seq:
        if word != 0:
            script += wordmap[word]
        if not char_level: script += "1"
    return script

def genSequence(epoch,logs):
    print()
    if verbose:
        print("Generating text for epoch %d: " % epoch, end="", flush=True)
    seed_text = np.zeros([1,seq_size])
    prediction = ""
    for  _  in range(sample_len):
        predict = model.predict_classes(seed_text, verbose = 0)
        #print(predict)
        text = sequence_to_text([predict[0]], plays_token, char_level)
        prediction += text
        seed_text = np.append(seed_text[:,1:],[predict], axis=1)
    if verbose:
        print("Done")
    print("-"*40 + 'Generated sequence' + '-'*40)
    print(prediction)
    print("-"*40 + 'Terminate sequence' + '-'*40)


path = 'saved_models/model_{epoch:02d}.h5'
checkpoint = callbacks.ModelCheckpoint(path, verbose=verbose)
reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=.5)
print_sample = callbacks.LambdaCallback(on_epoch_end=genSequence)
print_test = callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x_plays,y_plays,epochs=epochs,verbose=verbose, batch_size = batch_size, steps_per_epoch=steps_per_epoch, 
          validation_split=0.2, callbacks=[print_sample, checkpoint, reduceLR])


#print(kpp.genSequence(model, plays_token, seq_size=seq_size, char_level=char_level, verbose=verbose, sample_len=sample_len))