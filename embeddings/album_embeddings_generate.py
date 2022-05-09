import numpy as np
from PIL import Image
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, nz=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 2, stride=2, padding=0), # 64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 2, stride=2, padding=0), #32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 2, stride=2, padding=0), #16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 2, stride=2, padding=0), #8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 2, stride=2, padding=0), #4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, 2, stride=2, padding=0), #2 
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4096, nz)
        )
  
    def forward(self, x):
        return self.net(x)
class Decoder(nn.Module):
    def __init__(self, nz=128):
        super().__init__()
        self.map = nn.Linear(nz,4096)   # for initial Linear layer
        self.net = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, 1, stride=1, padding=0),
        )
  
    def forward(self, x):
        return self.net(self.map(x).reshape(-1, 1024, 2, 2))
class AutoEncoder(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.encoder = Encoder(nz)
        self.decoder = Decoder(nz)

    def forward(self, x):
        #print(x.shape)
        out = self.encoder(x)
        #print(out.shape)
        out = self.decoder(out)
        #print(out.shape)
        return self.decoder(self.encoder(x))

    def reconstruct(self, x):
        return self.forward(x)

BATCH_SIZE=128

data_all = np.load("../npy/normalized-image.npy")
print(data_all.shape)

dim = 256
ae_model = AutoEncoder(dim)
ae_model = nn.DataParallel(ae_model)
ae_model.load_state_dict(torch.load('model.p',map_location='cpu'))

data_load = torch.utils.data.DataLoader(torch.from_numpy(data_all).float(),batch_size=BATCH_SIZE)

embedding_np = np.zeros(shape=(len(data_all),dim))
for i, chunk in enumerate(data_load):
    embedding = ae_model.module.encoder(torch.from_numpy(np.array(chunk)).float())
    print(embedding.shape)
    np_sub = embedding.detach().cpu().numpy()
    embedding_np[i*BATCH_SIZE:i*BATCH_SIZE + len(np_sub)] = np_sub
    print(embedding_np.shape)

with open('../npy/album_embedding.npy', 'wb') as f:
    np.save(f, embedding_np)
