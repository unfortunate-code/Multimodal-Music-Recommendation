from fetch_lyrics import fetch_lyrics
import pickle
import time
import json

def get_lyrics():
    start = time.time()
    with open('../Data/MPD_Large/song_id2info.data', 'rb') as f:
        songs_dict = pickle.load(f)
    lyrics_dict = {}
    index = 0
    for _, s in songs_dict.items():
        artist_name = s['artist_name']
        song_name = s['track_name']
        try:
            lyrics = fetch_lyrics(song_name, artist_name)
        except:
            print("Failed fetching for " + song_name + " by " + artist_name)
        if lyrics:
            key = str(len(artist_name)) + '_' + artist_name + '_' + str(len(song_name)) + '_' + song_name
            lyrics_dict[key] = lyrics
        if len(lyrics_dict) == 1024:
            file_name = '../Data/lyrics/' + str(index) + '.json'
            with open(file_name, "w") as file:
                json.dump(lyrics_dict, file)
            lyrics_dict.clear()
            index += 1
    # Dump the rest.
    if len(lyrics_dict) > 0:
        file_name = '../Data/lyrics/' + str(index) + '.json'
        with open(file_name, "w") as file:
            json.dump(lyrics_dict, file)
    print(time.time() - start)

if __name__ == '__main__':
    get_lyrics()
