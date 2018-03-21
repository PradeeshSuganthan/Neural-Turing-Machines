import sys
sys.path.insert(0, './')
from keras.preprocessing import text
import keras_preprocessing as kpp

 x_plays, y_plays, plays_token = kpp.getPlays()
