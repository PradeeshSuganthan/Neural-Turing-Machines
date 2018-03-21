import numpy as np
import keras_preprocessing as kpp

from keras.preprocessing import text

x_plays, y_plays, plays_token = kpp.getPlays()


play = kpp.sequence_to_text(x_plays, plays_token)

print(play[0])