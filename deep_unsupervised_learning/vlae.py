from torch.distributions.multivariate_normal import MultivariateNormal

from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.utils.data as data
import math


def quantize(x, n_bit):
    x = x * 0.5 + 0.5 # to [0, 1]
    x *= 2 ** n_bit - 1 # [0, 255] 
    x = torch.floor(x + 1e-4) # [0, 255]
    return x


class Encoder(nn.Module):
    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.Conv2d(channel_dim, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
        )
        self.out = nn.Linear(256 * 4 * 4, z_dim)
        self.made = MADE((z_dim//2,), 2, hidden_size=[512, 512])
        self.prior = torch.distributions.Normal(torch.tensor(0.).cuda(), torch.tensor(1.).cuda())

    def forward(self, x):
        x = x.float()
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        z = self.out(x)
        return z

    def reverse(self, epis):
        m = self.made(epis)
        mu, log_sigma = m[:, 0], m[:, 1]
        zs = (epis - mu) * torch.exp(-log_sigma)
        return zs


class Decoder(nn.Module):

    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.z_dim = z_dim
        self.channel_dim = channel_dim

        self.linear = nn.Linear(z_dim, 4 * 4 * 128)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, z):
        z = z.float()
        # x = nn.ReLU()(self.linear(z))
        x = self.linear(z)
        x = x.view(-1, 128, 4, 4) 
        output = self.main(x) #[128, 3, 32, 32]
        output = torch.tanh(output)
        return output


class VLAE(nn.Module):
    def __init__(self, z_dim, channel_dim):
        super().__init__()
        self.encoder = Encoder(2 * z_dim, channel_dim)
        self.decoder = Decoder(z_dim, channel_dim)
        self.z_dim = z_dim

    def loss(self, x):
        z = self.encoder(x)
        mu, log_sigma = z.chunk(2, dim=1)
        z = torch.randn_like(mu) * torch.exp(log_sigma) + mu
        m = self.encoder.made(z)
        epis_mu, epis_log_sigma = m[:, 0], m[:, 1]
        epis = z * torch.exp(epis_log_sigma) + epis_mu #[128, 32]
        recon = self.decoder(z)
        recon_loss = ((recon-x)**2).mean(dim = 0).sum()

        log_p_z = epis_log_sigma.mean(dim = 0).sum() + self.encoder.prior.log_prob(epis).mean(dim = 0).sum()
        log_q_z = Normal(mu, torch.exp(log_sigma)).log_prob(z).mean(dim = 0).sum()
        kl_loss = (log_q_z - log_p_z)
        return recon_loss + kl_loss, recon_loss, kl_loss

    def sample(self, num, z_dim):
        epis = np.random.normal(0, 1, num * z_dim)
        epis = epis.reshape((num, z_dim))
        epis = torch.from_numpy(epis).cuda().float()
        zs = self.encoder.reverse(epis)
        pred = self.decoder(zs)
        pred = quantize(pred, 8).detach().cpu().numpy()
        return pred.astype(int)

    def reconstruct(self, x):
        z = self.encoder(x)
        mu, log_sigma = z.chunk(2, dim=1)
        z = torch.randn_like(mu) * (log_sigma).exp() + mu
        recon = self.decoder(z)
        recon = quantize(recon, 8).detach().cpu().numpy()
        return recon

    def interp(self, samples1, samples2):
        z1 = self.encoder(samples1)
        mu, log_sigma = z1.chunk(2, dim=1)
        z1 = torch.randn_like(mu) * (log_sigma).exp() + mu
        z2 = self.encoder(samples2)
        mu, log_sigma = z2.chunk(2, dim=1)
        z2 = torch.randn_like(mu) * (log_sigma).exp() + mu
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

def normalize(data):
  return (225 * (data-data.min())/(data.max() - data.min())).astype(int)

