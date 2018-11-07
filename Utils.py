import numpy as np
from keras.models import load_model

class DataExtractor:

    def __init__(self, length, step, data):
        self.length = length
        self.step = step
        self.text = open(data, "r").read().lower()

        self.words = self.get_words()               # all the unique words that appear on the text
        #print(self.words)
        self.chars = self.get_chars()               # all chars that appear throught the text
        #print(self.chars)
        self.get_sequences()                        # assumig step 3, sequences array made each of ['abcdefghik.. to 40 chars' , 'defghik.. to 40 chars', ...]
        #print(len(self.seq))

        self.c_t_i, self.i_c = self.get_c_dicts()   # char dictionaries
        #print(self.c_t_i, "\n", self.i_c)
        self.word_2_vec()                           # X and Y fitting arrays

    #   Returns two dictionaries relating a char with an index in both ways
    def get_c_dicts(self):
        return dict((c, i) for i, c in enumerate(self.chars)) , dict((i, c) for i, c in enumerate(self.chars))

    #   Returns all the words in the test sample
    def get_words(self):
        words = set()
        for sentence in self.text.split("\n"):
            for word in sentence.split(" "):
                words.add(word)
        return words

    #   Gets all single characters in the text
    def get_chars(self):
        return sorted(list(set(self.text)))

    #   Sets all sequences (X-array) and Y-vector for target values
    def get_sequences(self):
        self.seq = []
        self.n_c = []
        for i in range(0, len(self.text) - self.length, self.step):
            self.seq.append(self.text[i:i+self.length])
            self.n_c.append(self.text[i + self.length])

    def word_2_vec(self):
        # Generates X and Y vectors for model fitting (1-hot arrays)
        self.X = np.zeros((len(self.seq), self.length, len(self.chars)), dtype=np.bool)
        self.Y = np.zeros((len(self.seq), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(self.seq):
            for t, char in enumerate(sentence):
                self.X[i, t, self.c_t_i[char]] = 1
            self.Y[i, self.c_t_i[self.n_c[i]]] = 1

    #   Load a previously saved model
    def load_model_saved(self, path):
        self.model = load_model(path)

    #   Export options for prediction
    def export_options(self):
        #options = LTSM_OPTIONS(self.length, self.chars, self.c_t_i, self.i_c, self.words)
        return LTSM_OPTIONS(self.length, self.chars, self.c_t_i, self.i_c, self.words)

#   Class for settings & options exporting to 
class LTSM_OPTIONS:
    def __init__(self, seq_size, chars, cti, ic, w):
        self.seq_size = seq_size
        self.chars = chars
        self.char_dict = cti
        self.i_c = ic
        self.words = w

