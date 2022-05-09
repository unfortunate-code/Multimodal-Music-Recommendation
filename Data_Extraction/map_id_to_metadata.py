from lyricsgenius import Genius
import pickle
import time

ACCESS_TOKEN = 'efdoQZPNfD67-vtu0L77LV6XYjVg4OepeLPxsun2Y8Ferpxvq-be7a8osNEpkvxZ'

def get_mappings():
    start = time.time()
    genius = Genius(ACCESS_TOKEN)
    genius.remove_section_headers = True
    genius.retries = 3
    with open('../Data/MPD_Large/song_id2info.data', 'rb') as f:
        songs_dict = pickle.load(f)
    map = {}
    for _, s in songs_dict.items():
        artist_name = s['artist_name']
        song_name = s['track_name']
        song = None
        try:
            song = genius.search_song(song_name, artist_name)
        except:
            print("Failed fetching for " + song_name + " by " + artist_name)
        if song:
            inner_map = {}
            inner_map['title'] = song.__dict__['_body']['full_title'].replace('\xa0',' ')
            inner_map['album_art'] = song.__dict__['_body']['header_image_thumbnail_url']
            inner_map['song_art'] = song.__dict__['_body']['song_art_image_thumbnail_url']
            map[song.__dict__['id']] = inner_map
    if len(map) > 0:
        file_name = '../Data/genius_song_metadata.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(map, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(time.time() - start)

if __name__ == '__main__':
    get_mappings()
