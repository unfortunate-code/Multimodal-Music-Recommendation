import json
import pickle

def filter_genres():
    with open('../Data_Extraction/genres_new.json') as file:
        total_map = json.load(file)
    with open('../Data/spotify_genius_id_mapping.pkl', "rb") as file:
        id_mapping = pickle.load(file)
    with open('../Data/genius_song_metadata.pkl', "rb") as file:
        meta_mapping = pickle.load(file)
    map = {}
    print(len(total_map))
    print(len(meta_mapping))
    index = 0
    for s_id, g_id in id_mapping.items():
        if g_id in meta_mapping:
            song = meta_mapping[g_id]['title']
            if song in total_map:
                map[song] = total_map[song]
            else:
                if index < 10:
                    print(song)
                    index += 1
    print(len(map))
    reverse_map = {}
    for song, genres in map.items():
        for genre in genres:
            reverse_map.setdefault(genre, [])
            reverse_map[genre].append(song)
    import heapq
    heap = []
    for genre in reverse_map.keys():
        heapq.heappush(heap, (-len(reverse_map[genre]), genre))
    final_genres = []
    seen = set()
    while len(seen) < len(map):
        tmp = heapq.heappop(heap)
        final_genres.append(tmp[1])
        seen |= set(reverse_map[tmp[1]])
        if len(seen) > 100000: print(len(final_genres))
    final_map = {}
    for genre in final_genres:
        for song in reverse_map[genre]:
            final_map.setdefault(song, [])
            final_map[song].append(genre)
    with open('../Data_Extraction/genres_filtered.json', 'w') as file:
        json.dump(final_map, file)
    print(len(final_genres))

if __name__ == '__main__':
    filter_genres()
