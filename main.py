### MAIN ###

import sys
import numpy as np
from tensorflow.keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import layers


# modules
import text_preparation as tp
# import lstm

# Params
seq_length = 100

# main
raw_text = tp.load_text()
raw_text = tp.clean_text(raw_text)
chars, char_to_int, n_vocab = tp.map_text(raw_text)
data_X, data_y = tp.get_train_seqs(raw_text, char_to_int, seq_length)
X, y = tp.get_X_y(data_X, data_y, seq_length, n_vocab)

assert len(data_y) == len(raw_text) - seq_length

int_to_char = dict((i, c) for i, c in enumerate(chars))


# lstm.py
def create_model(X, y):
    model = Sequential()
    # input_len = max_sequence_len - 1
    # model.add(Embedding(total_words, 10, input_length=input_len))
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return(model)


model = Sequential()
model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
model.add(LSTM(hidden_size, return_sequences=True))
model.add(LSTM(hidden_size, return_sequences=True))
if use_dropout:
    model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(vocabulary)))
model.add(Activation('softmax'))



class lstm_struct(): # keras.Model
    def __init__(self, width, depth, nb_classes):
        #super(Lenet_like, self).__init__(name='lenet')
        self.width = width # width: number of params in the first layer
        self.depth = depth # depth: number of hidden layers
        self.nb_classes = nb_classes # number of classes to predict

        self.conv2D_first = layers.Conv2D(filters=self.width,
                                          kernel_size=(3, 3),
                                          activation='relu',
                                          padding='same', # adds a line of pixels identical to the border pixels
                                          name='conv2D_first')
        self.conv2D = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')
        self.AveragePooling2D = layers.AveragePooling2D()
        self.Flatten = layers.Flatten()
        self.dense_1 = layers.Dense(64, activation='relu')
        self.dense_last = layers.Dense(nb_classes, activation='softmax')


    def __call__(self, input_tensor):
        _y = self.conv2D_first(input_tensor)
        _y = self.AveragePooling2D(_y)

        for d in range(self.depth - 1): # -1 because we already added a conv layer
            _y = self.conv2D(_y)
            _y = self.AveragePooling2D(_y)

        _y = self.Flatten(_y) # we come back to a vector (usual neural network with dense layers from now on)
        _y = self.dense_1(_y)
        _y = self.dense_last(_y)

        return(_y)


model = create_model(X, y)
model.summary()


# fit
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list, verbose=2)

# generate texts
# load the network weights
filename = "weights-improvement-20-2.5857.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# pick a random seed
def pick_random_seed(data_X, int_to_char):
    start = np.random.randint(0, len(data_X) - 1)
    pattern = data_X[start]
    print("Seed:")
    print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
    return(pattern)


def generate_text(seed_text, int_to_char, nb_next_chars=1000):
    # generate 1000 characters
    for i in range(nb_next_chars):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        seq_in = [int_to_char[value] for value in pattern]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone.")


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()



seed_text = pick_random_seed(data_X, int_to_char)
seed_text = generate_text(seed_text, int_to_char, model, max_sequence_len)




# Second attempt: add a layer
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# fit
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1)

model.fit(X, y, epochs=25, batch_size=128, callbacks=callbacks_list.append(early_stopping))


# pick a random seed
start = np.random.randint(0, len(data_X)-1)
pattern = data_X[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")


# generate characters
for i in range(200):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("\nDone.")


# # Tune the dropout parameter
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.1))
model.add(LSTM(256))
model.add(Dropout(0.1))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


model.fit(X, y, epochs=25, batch_size=128, callbacks=callbacks_list.append(early_stopping))


# # Epochs = 50 and batch_size=128
model.fit(X, y, epochs=50, batch_size=128, callbacks=callbacks_list.append(early_stopping))

## generate characters
for i in range(200):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")


# # Epoch=50 and smaller batch size=98
model.fit(X, y, epochs=50, batch_size=98, callbacks=callbacks_list.append(early_stopping))


# # Smaller batch size=92 and epochs=70
model.fit(X, y, epochs=70, batch_size=92, callbacks=callbacks_list.append(early_stopping))

## generate characters
for i in range(500):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("\nDone.")


# # Add more layer (too heavy)
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')


# # Predict fewer than 1000 characters for a given seed

# generate characters
for i in range(100):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print("\nDone.")
