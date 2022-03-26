from bs4 import BeautifulSoup
import requests
from fetch_lyrics import fetch_lyrics

def get_lyrics():
    file = open('../Data/unique_song_spotify.txt', 'r')
    lines = file.readlines()
    file.close()
    for i in range(len(lines)):
        s = lines[i].split('\t')
        artistname = s[0]
        songname = s[1]
        track_id = s[2]
        artist_id = s[3]
        lyrics = fetch_lyrics(songname, artistname)
        if lyrics:
            fileName = '../Data/lyrics/' + '_' + str(len(artistname)) + '_' + artistname + '_' + str(len(songname)) + '_' + songname + '_' + str(len(track_id)) + '_' + track_id + '_' + str(len(artist_id)) + '_' + artist_id
            file = open(fileName, 'w')
            file.write(lyrics)
            file.close()

if __name__ == '__main__':
    get_lyrics()
