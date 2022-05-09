import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LinearVAE(nn.Module):
    def __init__(self, latent_dim=150, beta=0):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.criterion = torch.nn.MSELoss()
        self.beta = beta
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        z, mu, log_var = self.encode(x) # encoding
        reconstruction = self.decoder(z) # decoding
        return reconstruction, mu, log_var
    
    def encode(self, x):
        q = self.encoder(x)
        q = q.reshape(-1, 2, self.latent_dim)
        mu = q[:, 0, :] # the first feature values as mean
        log_var = q[:, 1, :] # the other feature values as variance
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
    def loss(self, x, rec, mu, log_var):
        # compute reconstruction loss
        rec_loss = self.criterion(x, rec)
        # compute KL divergence loss
        log_sigma = 0.5*log_var
        mu_unit = torch.zeros_like(mu)
        log_sigma_unit = torch.zeros_like(log_sigma)
        kl_loss = kl_divergence(mu, log_sigma, mu_unit, log_sigma_unit)
        kl_loss = torch.sum(kl_loss,axis=1) # sum across the latent dimension, not the batch dimension
        kl_loss = torch.mean(kl_loss) # make sure that this is a scalar, not a vector / array 

        return rec_loss + self.beta * kl_loss, {'rec_loss': rec_loss.cpu().detach().numpy(), 'kl_loss': kl_loss.cpu().detach().numpy()}

class Encoder(nn.Module):
    def __init__(self, latent_dim=150):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_features=11, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=100),
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
            nn.Linear(in_features=100, out_features=100),
            nn.LeakyReLU(),
            nn.Linear(in_features=100, out_features=11),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x) 
    
def kl_divergence(mu1, log_sigma1, mu2, log_sigma2):
  """Computes KL[p||q] between two Gaussians defined by [mu, log_sigma]."""
  return (log_sigma2 - log_sigma1) + (torch.exp(log_sigma1) ** 2 + (mu1 - mu2) ** 2) \
               / (2 * torch.exp(log_sigma2) ** 2) - 0.5
