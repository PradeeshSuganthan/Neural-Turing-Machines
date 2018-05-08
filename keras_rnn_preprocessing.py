import glob
import os
import re
import h5py
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
import numpy as np

def import_files(text_files_location):
    text_file_names = glob.glob(text_files_location)
    text_list = []
    for text_file_name in text_file_names:
        text_file = open(text_file_name)
        text_list.append(text_file.read())
        text_file.close()
    return text_list

def get_data(seq_size, step, text_files_location="./shakespeare/*.txt"):
    files = import_files(text_files_location)
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

    return combined_plays, chars, vocabulary, char_indices, indices_char, x, y
