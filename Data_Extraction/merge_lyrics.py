import json
from os import listdir
from os.path import isfile, join
import pickle

def merge_lyrics():
    filePaths = [join('../Data/lyrics', f) for f in listdir('../Data/lyrics') if isfile(join('../Data/lyrics', f))]
    res = {}
    for filePath in filePaths:
        with open(filePath) as file:
            res.update(json.load(file))
    with open('../Data/lyrics/merged_lyrics.pkl', 'wb') as file:
        pickle.dump(res, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    merge_lyrics()
