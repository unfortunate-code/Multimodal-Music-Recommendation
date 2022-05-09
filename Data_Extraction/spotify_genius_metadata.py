from fetch_coverart import fetch_coverart
import pickle
import time
import json

def get_metadata():
    start = time.time()
    with open('../Data/MPD_Large/song_id2info.data', 'rb') as f:
        songs_dict = pickle.load(f)
    metadata_dict = {}
    index = 0
    for _, s in songs_dict.items():
        artist_name = s['artist_name']
        song_name = s['track_name']
        try:
            metadata = fetch_coverart(artist_name, song_name)
        except:
            print("Failed fetching for " + song_name + " by " + artist_name)
        if metadata:
            key = str(len(artist_name)) + '_' + artist_name + '_' + str(len(song_name)) + '_' + song_name
            metadata_dict[key] = metadata
        if len(metadata_dict) == 1024:
            file_name = '../Data/metadata/' + str(index) + '.json'
            with open(file_name, "w") as file:
                json.dump(metadata_dict, file)
            metadata_dict.clear()
            index += 1
    # Dump the rest.
    if len(metadata_dict) > 0:
        file_name = '../Data/metadata/' + str(index) + '.json'
        with open(file_name, "w") as file:
            json.dump(metadata_dict, file)
    print(time.time() - start)

if __name__ == '__main__':
    get_metadata()
