import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from models.vae import LinearVAE
from models.genre_classifier import EncoderFC
from genre_utils import *

feature_columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

def load_data(filepaths):
    data = []
    for filepath in filepaths:
        print(f'loading {filepath}..')
        # df = pd.read_csv('Data/audio/audio_features.txt',sep='\t')
        df = pd.read_csv(filepath,sep='\t')
        data.append(df)
    data = pd.concat(data).drop_duplicates()
    return data

def preprocessing_features(data):    
    feats = np.array(data[feature_columns],dtype=float)
    # features = (features - np.mean(features, axis=0))/np.std(features, axis=0)
    feats = (feats - np.min(feats, axis=0))/np.ptp(feats, axis=0)
    print(feats.shape, feats.dtype)
    return feats

def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred >= threshold, dtype=float)
    return {
        'accuracy': accuracy_score(y_true=target, y_pred=pred)
    }
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path',help='data file with spotify_id and genres')
    parser.add_argument('features_path',help='audio features file')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--epoch',type=int,default=5)
    args = parser.parse_args()
    print(args)

    # Load data
    print('Load data...')
    data = pd.read_csv(args.data_path,sep='\t')
    assert 'spotify_id' in data.columns
    assert 'genre' in data.columns, 'need genre column where each song genre is in a|b|c|d format'
    data.spotify_id.astype(str)
    feats_df = pd.read_csv(args.features_path,sep='\t')
    
    # preprocess features, genres
    feats_df.id.astype(str)
    feats_df = feats_df[['id'] + feature_columns]
    feats_df = feats_df.rename(columns={'id':'spotify_id'})
    data = merge_df(data, feats_df)
    feats = preprocessing_features(data) 
    # print(data)
    genres = get_genre_from_df(data, sep='|')
    genre_list = get_genre_list(genres)
    genre_onehot = create_multilabel_onehot(genres,genre_list)
    # print(len(genres),len(genre_list))
    # assert len(feats) == len(genre_onehot), f'{len(feats)} != {len(genre_onehot)}'
    
    batch_size = args.batch_size
    epoch = args.epoch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Load model
    # Load VAE encoder weights
    print('Load model...')
    model_vae = LinearVAE(latent_dim=150).to(device)
    model_vae.load_state_dict(torch.load('models/vae.p'))
    model_ft = EncoderFC(150, len(genre_list), model_vae.encoder.state_dict())
    model_ft.to(device)
    model_ft.train()
    
    data_loader_ft = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.Tensor(feats),torch.Tensor(genre_onehot)), batch_size=batch_size, shuffle=True, num_workers=1)
    opt = torch.optim.Adam(model_ft.parameters(),lr=1e-3)          # create optimizer instance
    criterion = torch.nn.BCELoss()

    train_it = 0
    results = []
    for ep in range(epoch):
        print(f'Epoch {ep+1}/{epoch}')
        preds = []
        trues = []
        for batch_x, batch_y in data_loader_ft:
            opt.zero_grad()
            output = model_ft.forward(batch_x.float().to(device))
            rec_loss = criterion(output, batch_y.float().to(device))
            rec_loss.backward()
            opt.step()
            
            preds.extend(output.cpu().detach().numpy())
            trues.extend(batch_y.cpu().detach().numpy())
            
            if train_it % 1000 == 0:
                print("It {}: Loss: {}".format(train_it, rec_loss))
            train_it += 1
        
        result = calculate_metrics(np.array(preds), np.array(trues))
        print(result['accuracy'])
    print("Done!")
    
    torch.save(model_ft.state_dict(), 'models/audio_ft.p')

    # embeddings = []
    # model.eval()
    # with torch.no_grad():
    #     data_loader = torch.utils.data.DataLoader(feats, batch_size=batch_size, shuffle=False, num_workers=4)
    #     for sample_img in data_loader:  
    #         preds = model.forward(sample_img.float().to(device))
    #         embeddings.append(preds.cpu().detach().numpy())
    # embeddings = np.concatenate(embeddings)
    # print(embeddings.shape)
            
    # pickle.dump(embeddings, open('Data/audio_embeddings.p','wb'))
