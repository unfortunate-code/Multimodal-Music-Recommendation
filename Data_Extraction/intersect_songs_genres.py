import json

def main():
    file = open('../Data/unique_song_spotify.txt', 'r')
    lines = file.readlines()
    file.close()
    for i in range(len(lines)):
        s = lines[i].split('\t')
        lines[i] = s[0].lower() + ' by ' + s[1].lower()
    file = open('../Data/genres.json', 'r')
    genres_map = json.loads(file.read())
    genres_map = {k.lower() : v for k, v in genres_map.items()}
    file.close()
    count = 0
    for line in lines:
        if line in genres_map:
            count += 1
    print(count, len(lines), len(genres_map))


if __name__ == '__main__':
    main()
