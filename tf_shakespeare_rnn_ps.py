
import argparse
import tensorflow as tf

import tf_preprocessing

"""
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')
"""

def main(argv):
    #args = parser.parse_args(argv[1:])

    #read in data
    train_data = tf_preprocessing.load_data()

    #get dictionary of words sorted by commonness
    dictionary, rev_dictionary = tf_preprocessing.create_dictionary(train_data)

    #create model


    #train model



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
