from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.models import load_model
import numpy as np
import difflib

class RNN_LTSM:
    def __init__(self,  seq_l, char_l, neurons=128, epochs=20, l_r = 0.01, path="", save = False):
        self.neurons = neurons
        self.epochs = epochs
        self.path = path
        self.save_model = save

        # Model Creation
        self.model = Sequential([
            LSTM(self.neurons, input_shape=(seq_l, char_l)),    # LSTM RNN Model
            Dense(char_l),                                      # output = activation(dot(input*weitghs) + bias)
            Activation('softmax')                               # activation function
        ])
        # Model compilation
        self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=l_r))

    def fit(self, X, Y):
        # Class method calls model method
        self.model.fit(X, Y, batch_size=self.neurons, nb_epoch=self.epochs)
        if self.save_model:
            self.model.save(self.path)

    def predict(self, X, v=False):
        res = self.model.predict(X, verbose=v)[0]
        #print(res)
        return res

    def load_model_saved(self, path):
        self.model = load_model(path)


    def get_char_index(self, p, t=1.0):
        p = np.log(np.asarray(p).astype('float64'))/t
        exp_preds = np.exp(p)
        p = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, p, 1)
        return np.argmax(probas)

    def create_song(self, query, lyrics_len, options):
        print("\n\n")
        query = query[0:options.seq_size].lower()
        res = ''
        res = res + query

        for i in range(lyrics_len):
            print(" {0:.2f}".format(i/lyrics_len*100),"%",end="\r")    # Displays progress
            x = np.zeros((1, options.seq_size, len(options.chars)))
            # for each char generates 1-hot array for seguence_length array
            for t, char in enumerate(query):
                x[0, t, options.char_dict[char]] = 1.

            pred = self.predict(x, v=False)
            n_i = self.get_char_index(pred, 1.0)
            next_letter = options.i_c[n_i]
            res += next_letter
            query = query[1:] + next_letter

        print("Processing result...\n")
        res = self.correct_res(res, options)
        res = res[0:lyrics_len]
        return res

    def correct_res(self, str, options):
        aux_str = str[:40]
        str = str[40:]
        res = ''
        aux = np.zeros(len(options.words), dtype=float)
        aux_words = list(options.words)
        for sentence in str.split("\n"):
            for word in sentence.split(" "):
                #aux = np.zeros(len(options.words), dtype=float)
                for i, w in enumerate(aux_words):
                    aux[i] = difflib.SequenceMatcher(None, word, w).ratio()
                res_w = aux_words[np.argmax(aux)]
                res = res + res_w + " "
            res = res + "\n"
        res = aux_str + res
        return res

