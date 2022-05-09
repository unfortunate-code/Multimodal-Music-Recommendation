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
        out = self.encoder(x)
        out = self.decoder(out)
        return self.decoder(self.encoder(x))

    def reconstruct(self, x):
        return self.forward(x)

def dataprocess(data):
    data = np.transpose(data, (0,3,1,2))
    mean = np.mean(data, axis=(1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
    std = np.std(data, axis=(1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
    data = (data-mean)/std
    return data

def reverse_process(data):
    max = np.max(data)
    min = np.min(data)
    data = ((data - min)*255/(max - min)).astype(np.uint8)
    return data

def train():
    epochs = 100
    learning_rate = 1e-3
    nz=256
    ae_model = AutoEncoder(nz)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        ae_model = nn.DataParallel(ae_model)
        ae_model.to(device)
    ae_model = ae_model.train()  

    import torch.optim as optim
    opt = optim.Adam(ae_model.parameters(), lr=learning_rate)         # create optimizer instance
    criterion = nn.MSELoss()    # create loss layer instance

    train_it = 0
    for ep in range(epochs):
        print("Run Epoch {}".format(ep))
        rec_loss = 0.0
        val_loss = 0.0
        for i, data_train in enumerate(data_load_train):
            inputs = data_train
            opt.zero_grad()
            outputs = ae_model(inputs.to(device=device))
            rec_loss = criterion(outputs, inputs.to(device=device))
            rec_loss.backward()
            opt.step()
            if i%100 == 0:
                print("{} batch reconstruction Loss: {}".format(i, rec_loss))
        print("Reconstruction Loss: {}".format(rec_loss))
        with torch.no_grad():  
            for i, data_val in enumerate(data_load_val):
                inputs = data_val
                outputs = ae_model(inputs.to(device=device))
                val_loss = criterion(outputs, inputs.to(device=device))
            print("Validation Loss: {}".format(val_loss))
        torch.save(ae_model.state_dict(), 'model.p')

    print("Done!")

def test():
    ae_model.load_state_dict(torch.load('model.p', map_location='cpu'))
    embedding = ae_model.encoder(torch.from_numpy(data_val).float()[0:3])
    pics = ae_model.decoder(embedding)
    np_arr = pics.cpu().detach().numpy()
    np_arr = np.transpose(np_arr, (0,2,3,1))
    np_arr = reverse_process(np_arr)
    print(np_arr.shape)
    img = Image.fromarray(np_arr[2], 'RGB')
    img.save('my.png')

BATCH_SIZE=128
data_origin = np.load("npy/image-album.npy",allow_pickle=True)
print("shuffle ing...")
np.random.shuffle(data_origin)

print(data_origin.shape)
data_all = dataprocess(data_origin)
print(data_all.shape)
data_train, data_val = data_all[:60000],data_all[60000:]

data_load_train = torch.utils.data.DataLoader(torch.from_numpy(data_train).float(),batch_size=BATCH_SIZE,shuffle=True, num_workers=0)
data_load_val = torch.utils.data.DataLoader(torch.from_numpy(data_val).float(),batch_size=BATCH_SIZE,shuffle=True, num_workers=0)

train()
