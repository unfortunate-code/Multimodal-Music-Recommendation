import json
from lyricsgenius import Genius
import sys

ACCESS_TOKEN = 'efdoQZPNfD67-vtu0L77LV6XYjVg4OepeLPxsun2Y8Ferpxvq-be7a8osNEpkvxZ'

def fetch_genres(genresFileName, outputFileName):
    with open(genresFileName) as file:
        genres = [line.strip() for line in file]

    genius = Genius(ACCESS_TOKEN)
    genius.remove_section_headers = True
    genius.excluded_terms = ["(Remix)", "(Live)"]
    genius.retries = 3

    res = {}
    for genre in genres:
        genre = genre.lower()
        genre = genre.replace("'", "")
        genre = genre.replace("(", "")
        genre = genre.replace(")", "")
        genre = genre.replace('&', '-')
        genre = genre.replace(' ', '-')
        if len(genre) == 0: continue
        page = 1
        while page:
            try:
                out = genius.tag(genre, page=page)
            except Exception:
                print("Failed fetching", genre, page)
                break
            for hit in out['hits']:
                title_with_artists = hit['title_with_artists']
                res.setdefault(title_with_artists, [])
                res.get(title_with_artists).append(genre)
            page = out['next_page']

    with open(outputFileName, "w") as file:
        json.dump(res, file)

def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print("Run it as python fetch_genres.py genresFileName outputFileName")
        return
    fetch_genres(args[0], args[1])

if __name__ == '__main__':
    main()
