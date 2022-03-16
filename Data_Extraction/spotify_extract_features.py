import pandas as pd
import spotipy
from tqdm import tqdm
from spotipy.oauth2 import SpotifyClientCredentials

def batch_audio_features(spotify, ids):
    features = spotify.audio_features(','.join(ids)) # 100 IDs at a time
    return features

def get_audio_features(spotify, df):
    features = []
    for i in tqdm(range(0,len(df),100)):
        print(i)
        batch_feats = batch_audio_features(spotify, df['spotify_id'][i:i+100])
        features.extend(batch_feats)
    return features

if __name__ == '__main__':
    path = '../Data'
    
    print('loading ids.txt...',end=' ')
    df = pd.read_csv(path+'/ids.txt',sep='\t')
    print(len(df),'entries found')
    
    print('getting audio features...')
    df.dropna(inplace=True)
    spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials()) 
    features = get_audio_features(spotify, df)
    features_df = pd.DataFrame(features)
    features_df.to_csv(path+'/audio_features.txt',sep='\t',index=False)