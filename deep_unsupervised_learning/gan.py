from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.utils.data as data
import math

class D_2d(nn.Module):
    def __init__(self, channel_dim, feature_dim):
        super(D_2d, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(channel_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        output = self.main(x)
        return output

class G_2d(nn.Module):
    def __init__(self, channel_dim, feature_dim):
        super(G_2d, self).__init__()
        self.latent_dim = 1
        self.channel_dim = channel_dim
        self.main = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LeakyReLU(),
            nn.Linear(feature_dim, channel_dim),
        )

    def forward(self, x):
        output = self.main(x)
        output = nn.Tanh()(output)
        return output

    def loss(self, x):
        bs = x.shape[0]
        z = torch.Variable(torch.from_numpy(np.random.normal(0, 1, (bs, self.latent_dim)))).cuda()
        return self.cross_entropy(self.D(x), self.D(self(z)))

    def sample(self, n):
        z = torch.randn(n, self.latent_dim).cuda()
        out = self(z)
        return out


class GAN_2d(nn.Module):
    def __init__(self, D, G, mode=None):
        super(GAN_2d, self).__init__()
        self.D = D
        self.G = G
        self.cross_entropy = nn.BCELoss()
        self.mode = mode
        

    def D_loss(self, x):
        x = x.float()
        bs = x.shape[0]
        samples = self.G.sample(bs)

        pred = torch.cat((self.D(samples), self.D(x)), dim=0)
        labels = torch.cat((torch.zeros(bs), torch.ones(bs)), dim=0).cuda()
        return F.binary_cross_entropy(pred, labels)

    def G_loss(self, x):
        x = x.float()
        bs = x.shape[0]

        recon = self.D(self.G.sample(bs))
        labels = torch.ones(bs).cuda()
        return F.binary_cross_entropy(recon, labels)

    def loss(self, x):
      return -self.D_loss(x)

def train(train_data):
  """
  train_data: An (20000, 1) numpy array of floats in [-1, 1]

  Returns
  - a (# of training iterations,) numpy array of discriminator losses evaluated every minibatch
  - a numpy array of size (5000,) of samples drawn from your model at epoch #1
  - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
  - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid) 
      at each location in the previous array at epoch #1

  - a numpy array of size (5000,) of samples drawn from your model at the end of training
  - a numpy array of size (1000,) linearly spaced from [-1, 1]; hint: np.linspace
  - a numpy array of size (1000,), corresponding to the discriminator output (after sigmoid) 
      at each location in the previous array at the end of training
  """
  
  _, channel_dim = train_data.shape
  feature_dim = 256
  betas=(0, 0.9)
  lr = 1e-4
  epochs = 5
  """model"""
  D = D_2d(channel_dim, feature_dim).cuda()
  G = G_2d(channel_dim, feature_dim).cuda()
  model = GAN_2d(D, G).cuda()

  initial_samples = G.sample(5000).detach().cpu().numpy()
  lins = np.linspace(-1, 1, 1000).reshape((-1, 1))
  initial_lins = G(torch.from_numpy(lins).cuda().float())
  initial_output = D(initial_lins).detach().cpu().numpy()

  """training"""
  train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
  d_losses, g_losses = train_epochs(model, train_loader, dict(epochs=epochs, lr=lr, betas=betas))

  final_samples = G.sample(5000).detach().cpu().numpy()
  final_lins = G(torch.from_numpy(lins).cuda().float())
  final_output = D(final_lins).detach().cpu().numpy()
  final_lins, initial_lins = final_lins.detach().cpu().numpy(), initial_lins.detach().cpu().numpy()

  return d_losses, initial_samples, initial_lins, initial_output, final_samples, final_lins, final_output

