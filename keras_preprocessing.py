import glob
from keras.preprocessing import text

def import_files(text_files_location="./shakespeare/*.txt"):
    text_file_names = glob.glob(text_files_location)
    text_list = []
    for text_file_name in text_file_names:
        text_file = open(text_file_name)
        text_list.append(text_file.read())
        text_file.close()
    return text_list
    
def getPlays(lower=True, char_level=False, mode="binary"):
    plays_files = import_files()
    plays_token = text.Tokenizer(lower=lower, char_level=char_level)
    plays_token.fit_on_texts(plays_files)
    plays_sequenced = plays_token.texts_to_sequences(plays_files)
    y_plays = []
    x_plays = []
    for play in plays_sequenced:
        y_play = [0] + play
        x_play = play+ [0]
        y_plays.append(y_play)
        x_plays.append(x_play)
    return x_plays, y_plays, plays_token
