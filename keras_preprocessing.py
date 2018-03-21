import glob
from keras.preprocessing import text
from keras.utils import to_categorical
import numpy as np

def import_files(text_files_location="./shakespeare/*.txt"):
    text_file_names = glob.glob(text_files_location)
    text_list = []
    for text_file_name in text_file_names:
        text_file = open(text_file_name)
        text_list.append(text_file.read())
        text_file.close()
    return text_list
    
def getPlaysAsListOfSequences(lower=True, char_level=False):
    plays_files = import_files()
    plays_token = text.Tokenizer(lower=lower, char_level=char_level)
    plays_token.fit_on_texts(plays_files)
    plays_sequenced = plays_token.texts_to_sequences(plays_files)
    sequences = list()
    seq_size=1
    
    for play_sequenced in plays_sequenced:
        for i in range(1, len(play_sequenced)):
            start = i-seq_size
            if(start<0):
                start = 0
            sequence = play_sequenced[start:i+1]
            sequences.append(sequence)
            
    sequence_array = np.array(sequences)
    x_plays = sequence_array[:,0:-2]
    y_plays = sequence_array[:,-1]
    return x_plays, y_plays, plays_token


def sequence_to_text(seq, token):
    wordmap = {v: k for k,v in token.word_index.items()}
    plays = []
    for play in seq:
        script = ""
        for word in play:
            if word == 0:
                break
            else:
                script += wordmap[word] + " "
        plays.append(script)
    return plays
