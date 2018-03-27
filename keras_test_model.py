import keras_preprocessing as kpp
from keras.models import load_model

#global constants
seq_size=30
sample_len=5000
char_level=True
verbose=1
batch_size=512
epochs=4
steps_per_epoch = None
model_path = 'saved_models/model_04.h5'


x_plays, y_plays, plays_token = kpp.getPlaysAsListOfSequences(seq_size=seq_size, char_level=char_level, verbose=verbose)


model = load_model(model_path)

print(kpp.genSequence(model, plays_token, seq_size=seq_size, char_level=char_level, verbose=verbose, sample_len=sample_len))