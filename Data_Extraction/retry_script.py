from fetch_lyrics import fetch_lyrics
import json

def get_lyrics():
    with open('../Data/lyrics_to_retry.txt', 'r') as f:
        lines = f.readlines()
    lyrics_dict = {}
    for s in lines:
        s = s.split("#")
        artist_name = s[1].strip()
        song_name = s[0].strip()
        try:
            lyrics = fetch_lyrics(song_name, artist_name)
        except:
            print("Failed fetching for " + song_name + " by " + artist_name)
        if lyrics:
            key = str(len(artist_name)) + '_' + artist_name + '_' + str(len(song_name)) + '_' + song_name
            lyrics_dict[key] = lyrics
    if len(lyrics_dict) > 0:
        file_name = '../Data/lyrics/7100.json'
        with open(file_name, "w") as file:
            json.dump(lyrics_dict, file)

if __name__ == '__main__':
    get_lyrics()
