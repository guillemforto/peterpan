#########################################
#####        TEXT PREPARATION       #####
#########################################

import numpy as np
import string
from keras.utils import np_utils

def load_text():
    # Load ascii text and covert to lowercase
    filename = "peterpan.txt"
    raw_text = open(filename, 'r', encoding='utf-8').read()

    # Cutting the header and footer
    raw_text = raw_text[raw_text.find('All children, except one, grow up.'):]
    raw_text = raw_text[:raw_text.index('\n\n\nTHE END')]
    return(raw_text)


def clean_text(txt):
    ponct = string.punctuation.replace(".", "") # we keep sentences
    ponct = ponct.replace(",", "")
    txt = "".join(v for v in txt if v not in ponct).lower()
    txt = txt.encode("utf8").decode("ascii",'ignore')
    return txt


def map_text(raw_text):
    """Create a mapping of unique chars to integers"""
    print("Book length:", len(raw_text), 'characters')
    chars = sorted(list(set(raw_text)))
    n_vocab = len(chars)
    print("Number of unique characters:", n_vocab)
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    return chars, char_to_int, n_vocab


def get_train_seqs(raw_text, char_to_int, seq_length):
    """Prepare the dataset of (input, output) pairs, encoded as integers"""
    data_X = []
    data_y = []
    for i in range(0, len(raw_text) - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        data_X.append([char_to_int[char] for char in seq_in])
        data_y.append(char_to_int[seq_out])
    return data_X, data_y


def get_X_y(data_X, data_y, seq_length, n_vocab):
    """Reshape, normalize and one-hot-encode"""
    X = np.reshape(data_X, (len(data_X), seq_length, 1)) # reshape X: (251068, 100) => (251068, 100, 1)
    X = X / float(n_vocab) # normalize to [0,1]-range in order to use sigmoid later on
    y = np_utils.to_categorical(data_y) # one-hot-encode the output variable
    return X, y
