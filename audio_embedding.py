import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

from models.vae import LinearVAE

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepaths',nargs='+')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--epoch',type=int,default=5)
    args = parser.parse_args()
    print(args)

    feats_df = load_data(args.filepaths)
    feats = preprocessing_features(feats_df)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    batch_size = args.batch_size
    epoch = args.epoch

    data_loader = torch.utils.data.DataLoader(feats, batch_size=batch_size, shuffle=True, num_workers=4)
    model = LinearVAE(latent_dim=150).to(device)
    opt = torch.optim.Adam(model.parameters(),lr=1e-3)          # create optimizer instance
    criterion = torch.nn.MSELoss()

    train_it = 0
    for ep in range(epoch):
        print(f'Epoch {ep+1}/{epoch}')
        for sample_img in data_loader:
            # print(sample_img.float().shape)
            opt.zero_grad()
            output = model.forward(sample_img.float().to(device))
            rec_loss = criterion(output, sample_img.float().to(device))
            rec_loss.backward()
            opt.step()
            
            if train_it % 100 == 0:
                print("It {}: Reconstruction Loss: {}".format(train_it, rec_loss))
            train_it += 1
    print("Done!")
    
    torch.save(model.state_dict(), 'models/vae.p')
