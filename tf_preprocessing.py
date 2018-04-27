import glob
import os
import re
import h5py
import collections
import pandas as pd
import tensorflow as tf

train_path = "./shakespeare/*.txt"

def load_data(y_name='Species'):
    """Returns the shakespeare data as train_data."""
    text_file_names = glob.glob(train_path)
    texts = ""
    for text_file_name in text_file_names:
        text_file = open(text_file_name)
        texts+= text_file.read()
        text_file.close()

    #train_data = tf.data.TextLineDataset(text_files)

    #add .split() to switch from char to word model
    return texts

def create_dictionary(train_data):
    dictionary = {}
    count = collections.Counter(train_data).most_common()
    for word,_ in count:
        dictionary[word] = len(dictionary)
    rev_dictionary = dict(zip(dictionary.values(),dictionary.keys()))

    return dictionary,rev_dictionary