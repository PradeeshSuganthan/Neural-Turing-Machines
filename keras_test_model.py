import keras_rnn_preprocessing as kpp
import numpy as np
import random
import sys
from keras.models import load_model

#global constants
seq_size=40
step = 3
sample_len=5000
verbose=1
model_path = 'saved_models/nieztsche_50.h5'
nietzsche = "./nietzsche/*.txt"

files = kpp.import_files(nietzsche)
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


model = load_model(model_path)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


print()
start_index = 0#random.randint(0, len(combined_plays) - seq_size - 1)
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
