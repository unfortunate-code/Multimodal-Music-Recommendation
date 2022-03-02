from lyricsgenius import Genius
import sys

ACCESS_TOKEN = 'efdoQZPNfD67-vtu0L77LV6XYjVg4OepeLPxsun2Y8Ferpxvq-be7a8osNEpkvxZ'

def fetch_lyrics(songName = None, artist = None, fileName = None):
    genius = Genius(ACCESS_TOKEN)
    genius.remove_section_headers = True
    genius.excluded_terms = ["(Remix)", "(Live)"]
    if songName == None and artist == None:
        print("Pass at least one of song name and artist")
        return
    if songName == None and artist != None:
        artist = genius.search_artist(artist)
        artist.save_lyrics(fileName)
    elif songName != None and artist == None:
        song = genius.search_song(songName)
        print(song.lyrics)
    elif songName != None and artist != None:
        song = genius.search_song(songName, artist)
        print(song.lyrics)

def main():
    args = sys.argv[1:]
    artist = None
    songName = None
    fileName = None
    i = 0
    while i < len(args):
        if args[i] == '-a':
            artist = args[i + 1]
        elif args[i] == '-s':
            songName = args[i + 1]
        elif args[i] == '-f':
            fileName = args[i + 1]
        else:
            print("%s is not a valid argument. Use -a for artist, -s for song and -f for filename you want to save the lyrics in when only artist is passed", args)
            i += 1
            continue
        i += 2
    if songName == None and artist == None:
        print("Atleast one of artist and song needs to be passed. Use -s <song> -a <artist> -f <filename>")
        return
    fetch_lyrics(songName, artist, fileName)

if __name__ == '__main__':
    main()
