import numpy as np
from PIL import Image
import torch
import torch.nn as nn

nz = 256

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

def kl_divergence(mu1, log_sigma1, mu2, log_sigma2):
  """Computes KL[p||q] between two Gaussians defined by [mu, log_sigma]."""
  return (log_sigma2 - log_sigma1) + (torch.exp(log_sigma1) ** 2 + (mu1 - mu2) ** 2) \
               / (2 * torch.exp(log_sigma2) ** 2) - 0.5

class VAE(nn.Module):
  def __init__(self, nz, beta=1.0):
    super().__init__()
    self.beta = beta
    self.encoder = Encoder(2*nz)
    self.decoder = Decoder(nz)

  def forward(self, x):
    q = self.encoder(x) 
    mean = q[:,:nz]
    log_sigma = q[:,nz:]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    tmp = torch.from_numpy(np.random.normal(size=mean.size()).astype(np.float32)).to(device=device)
    z = torch.exp(0.5*log_sigma) * tmp
    z = mean + z    
    reconstruction = self.decoder(z)  

    return {'q': q, 'rec': reconstruction}

  def loss(self, x, outputs):
    mse = nn.MSELoss()
    rec_loss = mse(x, outputs['rec'])
    q=outputs['q']
    mean = q[:,:nz]
    log_sigma = q[:,nz:]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    kl_loss = kl_divergence(mean, log_sigma, torch.zeros(BATCH_SIZE).to(device=device), torch.zeros(BATCH_SIZE).to(device=device))
    kl_loss =  torch.mean(kl_loss)  
    return rec_loss + self.beta * kl_loss, \
           {'rec_loss': rec_loss, 'kl_loss': kl_loss}
    
  def reconstruct(self, x):
    q = self.encoder(x)
    reconstruction = self.decoder(q[:,:nz])
    return reconstruction


def train():
    learning_rate = 1e-3
    epochs = 100
    beta = 0.15

    # build VAE model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
    vae_model = VAE(nz, beta) #.to(device)    # transfer model to GPU if available
    vae_model.to(device)
    # ae_model.load_state_dict(torch.load('model.p'))
    vae_model = vae_model.train()  


    import torch.optim as optim
    opt = optim.Adam(vae_model.parameters(), lr=learning_rate)

    train_it = 0

    for ep in range(epochs):
        print("Run Epoch {}".format(ep))
        for i, data in enumerate(data_load_train):
            inputs = data
            opt.zero_grad()
            outputs = vae_model(inputs.to(device=device))
            total_loss, losses = vae_model.loss(inputs.to(device=device), outputs)
            total_loss.backward()
            opt.step()
            losses['rec_loss'] = losses['rec_loss'].cpu().detach().numpy()
            losses['kl_loss'] = losses['kl_loss'].cpu().detach().numpy()
            if i % 100 == 0:
                print("batch {}: Total Loss: {}, \t Rec Loss: {},\t KL Loss: {}"\
                    .format(train_it, total_loss, losses['rec_loss'], losses['kl_loss']))
            train_it += 1
            print("batch {}: Total Loss: {}, \t Rec Loss: {},\t KL Loss: {}"\
                .format(train_it, total_loss, losses['rec_loss'], losses['kl_loss']))
        with torch.no_grad():  
            for i, data in enumerate(data_load_val):
                inputs = data
                outputs = vae_model(inputs.to(device=device))
                total_loss, losses = vae_model.loss(inputs.to(device=device), outputs)
                losses['rec_loss'] = losses['rec_loss'].cpu().detach().numpy()
                losses['kl_loss'] = losses['kl_loss'].cpu().detach().numpy()
                print("batch {}: Total Loss: {}, \t Rec Loss: {},\t KL Loss: {}"\
                    .format(train_it, total_loss, losses['rec_loss'], losses['kl_loss']))
        torch.save(vae_model.state_dict(), 'model-vae.p')
    print("Done!")


BATCH_SIZE=128
data_origin = np.load("../npy/image-album.npy",allow_pickle=True)
print("shuffle ing...")
np.random.shuffle(data_origin)

print(data_origin.shape)
data_all = dataprocess(data_origin)
print(data_all.shape)
data_train, data_val = data_all[:60000],data_all[60000:]

data_load_train = torch.utils.data.DataLoader(torch.from_numpy(data_train).float(),batch_size=BATCH_SIZE,shuffle=True, num_workers=0)
data_load_val = torch.utils.data.DataLoader(torch.from_numpy(data_val).float(),batch_size=BATCH_SIZE,shuffle=True, num_workers=0)


train()
