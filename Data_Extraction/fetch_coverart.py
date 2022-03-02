from lyricsgenius import Genius
import sys

ACCESS_TOKEN = 'efdoQZPNfD67-vtu0L77LV6XYjVg4OepeLPxsun2Y8Ferpxvq-be7a8osNEpkvxZ'

def fetch_coverart(artist = None, songName):
    genius = Genius(ACCESS_TOKEN)
    genius.remove_section_headers = True
    genius.excluded_terms = ["(Remix)", "(Live)"]
    if songName == None:
        print("Song name cannot be None")
        return
    if songName != None and artist == None:
        song = genius.search_song(songName)
    elif songName != None and artist != None:
        song = genius.search_song(songName, artist)
    song_dict = genius.song(song.id)
    print("song_img: " + song_dict['song']['header_image_url'] + ", album_img: "  + song_dict['song']['song_art_image_url'] + ", song_description: " + song_dict['song']['description']['plain'])


def main():
    args = sys.argv[1:]
    artist = None
    songName = None
    i = 0
    while i < len(args):
        if args[i] == '-a':
            artist = args[i + 1]
        elif args[i] == '-s':
            songName = args[i + 1]
        else:
            print("%s is not a valid argument. Use -a for artist, -s for song", args)
            i += 1
            continue
        i += 2
    if songName == None:
        print("Song name needs to be passed. Use -s <song> -a <artist>")
        return
    fetch_coverart(artist, songName)

if __name__ == '__main__':
    main()
