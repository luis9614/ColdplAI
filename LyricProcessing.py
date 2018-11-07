from pandas import  read_csv, DataFrame
import numpy as np
from os import listdir
from nltk.tokenize import RegexpTokenizer
import codecs


class LyricExtractor:
    def __init__(self, dataset, indexes = [True, True, False, True]):
        self.query = ""
        self.data = read_csv("data/" + dataset, sep=",").values
        np.set_printoptions(threshold=np.inf)
        self.artists = artists = np.unique(self.data[:, 0])
        self.indexes = indexes

    def get_artist_file(self, artist_name):
        #tokenizer = RegexpTokenizer(r'\w+')
        #tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
        oned = sum(self.artists == artist_name)
        if oned == 1:
            lyrics = self.data[:, 0] == artist_name
            aux = self.data[lyrics, :]
            aux = aux[:, self.indexes]
            #print(aux)
            df = DataFrame(aux)
            query = artist_name.lower()
            query = query.replace(" ", "_")
            df.to_csv("data/byArtist/" + query + "_lyrics.csv")

            r = aux.size

            file = open("data/byArtist/" + query + "_lyrics.txt", "w")

            for i in range(r):
                str = np.array2string(aux[i])
                #str = str(str, 'utf-8')
                arr = str.split('\n')
                #with code
                for word in arr:
                    file.write(word)
                    file.write(" ")
                file.write("\n\n")
            file.close()
        else:
            print("Artist '", artist_name, "´not found.")

    def process_txt(self, filename):
        query = filename.lower()
        query = query.replace(" ", "_")
        filename = "data/byArtist/"+query + "_lyrics.txt"
        text = ""
        with codecs.open(filename, 'r', encoding='utf8') as f:
            text = f.read()
            print(text)
            text.replace("\\n", "\n")
        with codecs.open(filename, 'w', encoding='utf8') as f:
            f.write(text)

    def get_artist_file_txt(self, artist_name):
        oned = sum(self.artists == artist_name)
        if oned == 1:
            lyrics = self.data[:, 0] == artist_name
            aux = self.data[lyrics, :]
            aux = aux[:, self.indexes]
            df = DataFrame(aux)
            query = artist_name.lower()
            query = query.replace(" ", "_")
            df.to_csv("data/byArtist/" + query + "_lyrics.csv")
        else:
            print("Artist '", artist_name, "´not found.")

    def get_artists(self):
        files = listdir("data/byArtist")
        self.artist_dict = dict()
        aux_files = []
        for file in files:
            if file != '.DS_Store':
                new_name = file.replace("_lyrics.csv","").replace("_", " ")
                aux_files.append(new_name)
                self.artist_dict[new_name] = file
        return aux_files



de = LyricExtractor("lyrics.csv", indexes=[False, False, False, True])
#de.get_lyric_file("abba_lyrics.csv")
while True:
    query = input("artist name:\n")
    de.get_artist_file(query)
    de.process_txt(query)

arr = de.get_artists()
print(de.artist_dict)