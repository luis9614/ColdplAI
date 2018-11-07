from Utils import DataExtractor
from LSTM_Model import RNN_LTSM

# Behaviour Settings
LOAD_MODEL = True
MODEL_PATH = "models/pop_10e_40_seq_256neurons.h5"
SOURCE_LYRICS_PATH = "data/byArtist/ed_sheeran_lyrics.txt"
RESULT_PATH = "genlyrics/final.txt"

# Lyrics Settings
LYRYCS_LEN = 1500
L_SEQ = 40


data = DataExtractor(L_SEQ, 3, SOURCE_LYRICS_PATH)

model = RNN_LTSM(data.length, len(data.chars), neurons=256, epochs=10, path=MODEL_PATH, save=True)

if LOAD_MODEL:
    model.load_model_saved(MODEL_PATH)
else:
    model.fit(data.X, data.Y)


while True:
    query = input("\nTell me your inspiration\n")
    res = model.create_song(query, LYRYCS_LEN, data.export_options())
    #print(res)
    file = open(RESULT_PATH, 'w')
    file.write(res)
    file.close()

    print('DONE')
