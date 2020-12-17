
from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.utils.data as data
import math

class Encoder(nn.Module):
    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Conv2d(channel_dim, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
        )
        self.out = nn.Linear(256 * 4 * 4, z_dim)

    def forward(self, x):
        x = x.float()
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        # x = nn.Tanh()(x)
        # x = torch.clamp(x, -1, 1)
        return x


class Decoder(nn.Module):

    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.z_dim = z_dim
        self.channel_dim = channel_dim

        self.linear = nn.Linear(z_dim, 4 * 4 * 128)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, z):
        z = z.float()
        x = nn.ReLU()(self.linear(z)) #[128, 2048]
        x = x.view(-1, 128, 4, 4) 
        output = self.main(x) #[128, 3, 32, 32]
        # output = nn.Tanh()(output)
        # output = torch.clamp(output, -1, 1)
        return output

def quantize(x, n_bit):
    x = x * 0.5 + 0.5 # to [0, 1]
    x *= 2 ** n_bit - 1 # [0, 255] 
    x = torch.floor(x + 1e-4) # [0, 255]
    return x


class VAE(nn.Module):
    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.encoder = Encoder(2 * z_dim, channel_dim)
        self.decoder = Decoder(z_dim, channel_dim)
        self.z_dim = z_dim

    def loss(self, x):
        mu, log_var = self.encoder(x).chunk(2, dim=1)
        z = torch.randn_like(mu) * (0.5 * log_var).exp() + mu
        recon = self.decoder(z)
        recon_loss = ((recon-x)**2).mean(dim = 0).sum()
        kl_loss = 0.5 * (-log_var - 1 + torch.exp(log_var) + mu ** 2).mean(dim = 0).sum()
        return recon_loss + kl_loss, recon_loss, kl_loss

    def sample(self, num, z_dim):
      zs = np.random.normal(0, 1, num * z_dim)
      zs = zs.reshape((num, z_dim))
      zs = torch.from_numpy(zs).cuda().float()
      pred = self.decoder(zs)
      pred = quantize(pred, 8).detach().cpu().numpy()
      return pred.astype(int)

    def reconstruct(self, x):
      mu, log_var = self.encoder(x).chunk(2, dim=1)
      z = torch.randn_like(mu) * (0.5 * log_var).exp() + mu
      recon = self.decoder(z)
      recon = quantize(recon, 8).detach().cpu().numpy()
      return recon

    def interp(self, samples1, samples2):
      mu, log_var = self.encoder(samples1).chunk(2, dim=1)
      z1 = torch.randn_like(mu) * (0.5 * log_var).exp() + mu
      mu, log_var = self.encoder(samples2).chunk(2, dim=1)
      z2 = torch.randn_like(mu) * (0.5 * log_var).exp() + mu
      to_invert = torch.zeros(samples1.shape[0], 10, *samples1.shape[1:])
      alphas = np.arange(10)/10
      for a in range(10):
        interp = alphas[a]
        z = interp * z1 + (1-interp) * z2
        x = self.decoder(z)
        to_invert[:, a] = x
      result = to_invert.view(10 * len(samples1), *samples1.shape[1:])
      return quantize(result, 8).detach().cpu().numpy()

def preprocess(data):
  data = data.astype(float)
  rand = np.random.uniform(0, 1, data.shape) 
  data += rand #[0, 256]
  data = data/128 #[0, 2]
  data -= 1 #[-1, 1]
  return data 
