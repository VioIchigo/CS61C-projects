from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.utils.data as data
import math


class VAE_2d(nn.Module):
    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(channel_dim, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 2 * z_dim),
                                     )

        self.decoder = nn.Sequential(nn.Linear(z_dim, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 2 * channel_dim),
                                     )
        
    def reconstruct(self, mean, log_var):
      std = torch.exp(log_var/2)
      return torch.randn_like(mean) * std + mean

    def loss(self, x):
        x = x.float()
        mu, log_var = self.encoder(x).chunk(2, dim=-1)
        z = self.reconstruct(mu, log_var)

        x_mu, x_log_var = self.decoder(z).chunk(2, dim=-1)
        recon_loss = -Normal(x_mu, torch.exp(x_log_var/2)).log_prob(x).mean(dim=0).sum()
        kl_loss = 0.5 * (-log_var - 1 + torch.exp(log_var) + mu ** 2).mean(dim=0).sum()
        return recon_loss + kl_loss, recon_loss, kl_loss

    def sample(self, num, dim, noise = True):
      zs = np.random.normal(0, 1, num * dim)
      zs = zs.reshape((num, dim))
      zs = torch.from_numpy(zs).cuda().float()
      mu, log_var = self.decoder(zs).chunk(2, dim=-1)
      if noise:
        # log_var = nn.Tanh()(log_var)
        pred = self.reconstruct(mu, log_var)
      else:
        pred = mu
      return pred.detach().cpu().numpy()

