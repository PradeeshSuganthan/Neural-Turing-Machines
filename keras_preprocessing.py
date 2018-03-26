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

    
    
def write_hdf5(dataDict, DataName, PPDataDir, verbose=False):
    if verbose:
        print("Saving processed data to"+PPDataDir+DataName+".hdf5"+" : ", end="", flush=True)
    if not os.path.isdir(PPDataDir):
        os.makedirs(PPDataDir)
    with h5py.File(PPDataDir+DataName+".hdf5", "a") as PPData:
        for name, data in dataDict.items():
            PPData[name] = data
    if verbose:
        print("Done")
        
        
        
def read_hdf5(DataName, PPDataDir, verbose=False):
    dataDict = dict()
    if verbose:
        print("Loading processed data form"+PPDataDir+DataName+".hdf5"+" : ", end="", flush=True)
    with h5py.File(PPDataDir+DataName+".hdf5", "r") as PPData:
        for name, data in PPData.items():
            dataDict[name] = data[:]
    if verbose:
        print("Done")
    return dataDict

    
    
def getPlaysAsListOfSequences(file_location="./shakespeare/*.txt", PPDataDir = "./processed_data/", verbose=False, lower=True, char_level=False, seq_size=1):
    savedDataName = re.sub(r'\W+', '', file_location)+'_cl'+str(int(char_level))+'_ss'+str(seq_size)
    if verbose:
        print("Loading and tokenizing data : ", end="", flush=True)
    plays_files = import_files(file_location)
    plays_token = text.Tokenizer(lower=lower, char_level=char_level)
    plays_token.fit_on_texts(plays_files)
    if verbose:
        print("Done")
        
        
    if(os.path.exists(PPDataDir +savedDataName+".hdf5")):
        if verbose:
            print("Found preprocessed data.")
        dataDict = read_hdf5(savedDataName, PPDataDir, verbose=verbose)
        x_plays = dataDict['x_plays']
        y_plays = dataDict['y_plays']
            
            
    else:
        if verbose:
            print("No preprocessed data found.")
        plays_sequenced = plays_token.texts_to_sequences(plays_files)
        sequences = list()
        if verbose:
            print("Splitting into seq : ", end="", flush=True)
        for play_sequenced in plays_sequenced:
            for i in range(1, len(play_sequenced)):
                start = i-seq_size
                if(start<0):
                    start = 0
                seq = play_sequenced[start:i+1]
                sequences.append(seq)
                
                
        if verbose:
            print("Done")  
            print("Splitting sequence into x and y : ", end="", flush=True)
        sequence_array = sequence.pad_sequences(sequences)
        x_plays = sequence_array[:,0:-1]
        y_plays = sequence_array[:,-1]
        if verbose:
            print("Done")
            
            
        dataDict = {'x_plays': x_plays, 'y_plays':y_plays}
        write_hdf5(dataDict, savedDataName, PPDataDir, verbose=verbose)
        
        
    return x_plays, y_plays, plays_token


def sequence_to_text(seq, token, char_level=False):
    wordmap = {v: k for k,v in token.word_index.items()}
    script = ""
    for word in seq:
        script += wordmap[word]
        if not char_level: script += "1"
    return script



def genSequence(model, token, sample_len = 500, seq_size=50, char_level=False, verbose=False):
    if verbose:
        print("Generating text : ", end="", flush=True)

    seed_text = np.zeros([1,seq_size])
    prediction = ""
    for  _  in range(sample_len):
        predict = model.predict_classes(seed_text, verbose = 0)
        text = sequence_to_text([predict[0]], token, char_level)
        prediction += text
        seed_text = np.append(seed_text[:,1:],[predict], axis=1)

    if verbose:
        print("Done")
    print("-"*40 + 'Generated sequence' + '-'*40)
    return prediction

