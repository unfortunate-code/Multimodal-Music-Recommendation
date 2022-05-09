import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss =criterion(output,y)
    loss.backward()
    optimizer.step()

    return loss, output

def val(model, x, y, optimizer, criterion):
    output = model(x)
    loss = criterion(output,y)
    return loss, output

# data
AUDIO_EMBEDDING = "npy/common_audio_embedding.npy"
ALBUM_EMBEDDING = "npy/common_album_embedding.npy"
audio_embeddings = np.load(AUDIO_EMBEDDING, allow_pickle=True)
album_embeddings = np.load(ALBUM_EMBEDDING, allow_pickle=True)
audio_embeddings = torch.FloatTensor(audio_embeddings)
album_embeddings = torch.FloatTensor(album_embeddings)
concat_embedding = torch.cat((audio_embeddings, album_embeddings), axis=1)
data_train, data_val = concat_embedding[:60000],concat_embedding[60000:]

BATCH_SIZE = 16
data_load_train = DataLoader(data_train,batch_size=BATCH_SIZE,shuffle=False)
data_load_val = DataLoader(data_val,batch_size=BATCH_SIZE,shuffle=False)


EPOCHS = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

model = NeuralNetwork().to(device)
criterion = nn.MSELoss()
optm = Adam(model.parameters(), lr = 0.001)

for epoch in range(EPOCHS):
    epoch_loss = 0
    correct = 0
    for bidx, batch in enumerate(data_load_train):
        x_train, y_train = batch[:,:128], batch[:,:128]
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        loss, predictions = train(model,x_train,y_train, optm, criterion)
    print('Epoch {} Loss : {}'.format((epoch+1),loss))

    with torch.no_grad():  
        for i, data in enumerate(data_load_val):
            x_train, y_train = batch[:,:128], batch[:,:128]
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            loss, predictions = val(model,x_train,y_train, optm, criterion)
        print('Val {} Loss : {}'.format((epoch+1),loss))
    
    torch.save(model.state_dict(), 'recover_model.p')
