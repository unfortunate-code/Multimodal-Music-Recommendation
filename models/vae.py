import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LinearVAE(nn.Module):
    def __init__(self, latent_dim=150):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
        z = self.encode(x)
        # decoding
        reconstruction = self.decoder(z)
        return reconstruction
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.reshape(-1, 2, self.latent_dim)
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        z = self.reparameterize(mu, log_var)
        return z
 
class Encoder(nn.Module):
    def __init__(self, latent_dim=150):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_features=11, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=latent_dim*2)
        )
    def forward(self, x):
        return self.net(x) 
    
 
class Decoder(nn.Module):
    def __init__(self, latent_dim=150):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=11),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x) 