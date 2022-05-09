import os
import requests
import pickle

def extract_images():
    with open('../Data/genius_song_metadata.pkl', "rb") as file:
        meta_mapping = pickle.load(file)
    with open('../Data/spotify_genius_id_mapping.pkl', "rb") as file:
        id_mapping = pickle.load(file)

    for s_id, g_id in id_mapping.items():
        if g_id in meta_mapping:
            song_url = meta_mapping[g_id]['song_art']
            if song_url:
                try:
                    response_song = requests.get(song_url)
                    f_ext = os.path.splitext(song_url)[-1]
                    f_name = '../Data/images/song_' + s_id + f_ext
                    with open(f_name, 'wb') as f:
                        f.write(response_song.content)
                    response_song.close()
                    album_url = meta_mapping[g_id]['album_art']
                except:
                    print("Failed fetching song art for " + s_id)
            if album_url:
                try:
                    response_album = requests.get(album_url)
                    f_ext = os.path.splitext(album_url)[-1]
                    f_name = '../Data/images/album_' + s_id + f_ext
                    with open(f_name, 'wb') as f:
                        f.write(response_album.content)
                    response_album.close()
                except:
                    print("Failed fetching album art for " + s_id)

if __name__ == '__main__':
    extract_images()
