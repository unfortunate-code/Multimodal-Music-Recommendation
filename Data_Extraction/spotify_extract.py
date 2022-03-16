import pandas as pd
import spotipy
import os
import sys
# import json
# import time
from tqdm import tqdm
from spotipy.oauth2 import SpotifyClientCredentials

def batch_request(spotify, queries, start, end):
    # time1 = time.time()
    results = []
    for q in tqdm(queries[start:end]):
        res = spotify.search(q=q, limit=1)
        results.append(res)
    # print('time:',time.time()-time1)
    return results

if __name__ == '__main__':
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    manual_start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None
    path = '../Data'
    print('loading unique_songs.txt...')
    uniq = pd.read_csv(path+'/unique_songs.txt',sep='\t',header=None)
    queries = list(uniq[0] + ' artist:' + uniq[1])
    
    # with open('spotify.json','a+') as f:
    #     data = json.load(f)
    # total = len(data)
    # start,end = total, total+batch
    
    # with open('ids.txt','a+') as f:
    #     total = f.readline()
    #     if len(total) == 0:
    #         total = 0
    #         f.write('0\n')
    #     else:
    #         total = int(total)
    
    out_path = path+'/ids.txt'
    print('loading',out_path,'...',end=' ')
    if not os.path.exists(out_path):
        with open(out_path,'w') as f:
            f.write('title\tartist\tid\tspotify_id\n')
    df = pd.read_csv(out_path,sep='\t')
    total = len(df)
    print(total,'entries found')
    
    
    start,end = total, total+batch
    if manual_start_idx:
        start,end = manual_start_idx,manual_start_idx+batch
    print('start,end =',start,end)
    
    print('calling Spotify API...')
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials()) 
    results = batch_request(spotify, queries, start, end)
    print(len(results))
    
    # print('appending results...')
    for i in range(len(results)):
        try:
            spotify_id = results[i]['tracks']['items'][0]['id']
        except IndexError:
            spotify_id = None
        title,artist,id = uniq.iloc[start+i]
        # df.loc[len(df.index)] = [title,artist,id]
        df = df.append({'title':title,'artist':artist,'id':id,'spotify_id':spotify_id}, ignore_index=True)
    print('New total entries',len(df))
    
    df.to_csv(out_path,sep='\t',index=False)
            
            

