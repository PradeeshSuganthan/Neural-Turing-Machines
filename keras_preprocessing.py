import glob
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
    
def getPlaysAsListOfSequences(file_location="./shakespeare/*.txt", lower=True, char_level=False, seq_size=1):
    plays_files = import_files(file_location)
    plays_token = text.Tokenizer(lower=lower, char_level=char_level)
    plays_token.fit_on_texts(plays_files)
    plays_sequenced = plays_token.texts_to_sequences(plays_files)
    sequences = list()
    
    for play_sequenced in plays_sequenced:
        for i in range(1, len(play_sequenced)):
            start = i-seq_size
            if(start<0):
                start = 0
            seq = play_sequenced[start:i+1]
            sequences.append(seq)
            
    sequence_array = sequence.pad_sequences(sequences)
    x_plays = sequence_array[:,0:-1]
    y_plays = sequence_array[:,-1]
    return x_plays, y_plays, plays_token


def sequence_to_text(seq, token):
    wordmap = {v: k for k,v in token.word_index.items()}
    script = ""
    for word in seq:
        script += wordmap[word] + " "
    return script


def genSequence(model, token, sample_len = 50):
    seed_text = [0,0,0]
    prediction = ""
    for _ in range(sample_len):
        print(seed_text)
        predict = model.predict_classes(seed_text, verbose = 0)
        print(predict)
        text = sequence_to_text([predict[0]], token)
        prediction += text
        seed_text = predict

    print("-"*40 + 'Generated sequence' + '-'*40)
    return prediction
