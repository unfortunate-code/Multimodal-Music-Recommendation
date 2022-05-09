from lyricsgenius import Genius
import pickle
import time

ACCESS_TOKEN = 'efdoQZPNfD67-vtu0L77LV6XYjVg4OepeLPxsun2Y8Ferpxvq-be7a8osNEpkvxZ'

def get_id_mappings():
    start = time.time()
    genius = Genius(ACCESS_TOKEN)
    genius.remove_section_headers = True
    genius.retries = 3
    with open('../Data/MPD_Large/song_id2info.data', 'rb') as f:
        songs_dict = pickle.load(f)
    id_mapping_dict = {}
    songs_mapping_dict = {}
    for id, s in songs_dict.items():
        artist_name = s['artist_name']
        song_name = s['track_name']
        song = None
        try:
            song = genius.search_song(song_name, artist_name)
        except:
            print("Failed fetching for " + song_name + " by " + artist_name)
        if song:
            key = str(len(artist_name)) + '_' + artist_name + '_' + str(len(song_name)) + '_' + song_name
            songs_mapping_dict[key] = id
            id_mapping_dict[id] = song.id
    if len(songs_mapping_dict) > 0:
        file_name = '../Data/spotify_genius_id_mapping.pkl'
        with open(file_name, "wb") as file:
            pickle.dump(id_mapping_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
        file_name = '../Data/song_spotify_id_mapping.pkl'
        with open(file_name, "wb") as file:
            pickle.dump(songs_mapping_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(time.time() - start)

if __name__ == '__main__':
    get_id_mappings()
