from torch.autograd import Variable
from deepul.hw2_helper import *
import deepul.pytorch_util as ptu
from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.utils.data as data
import math

class AffineLayer(nn.Module):
  def __init__(self, feature_dim, num_layers, nout):
    super().__init__()
    self.hidden_sizes = [feature_dim] * num_layers
    hs = [1] + self.hidden_sizes + [nout]
    net = []
    for h0, h1 in zip(hs, hs[1:]):
      net.extend([nn.Linear(h0, h1), nn.Tanh(),])
    net.pop()
    self.net = nn.Sequential(*net)
    self.scale = Variable(torch.zeros(()), requires_grad=True).cuda()
    self.shift = Variable(torch.zeros(()), requires_grad=True).cuda()

  def forward(self, x1, x2):
    z1 = x1
    x1 = x1.reshape((-1, 1))
    pred = self.net(x1)
    g_scale = pred[:, 0]
    g_scale = nn.Tanh()(g_scale)
    g_shift = pred[:, 1]
    log_scale = g_scale
    # :/ 口亨
    # log_scale = self.scale * g_scale + self.shift
    z2 = torch.exp(log_scale) * x2 + g_shift
    log_det = log_scale
    return z1, z2, log_det

class RealNVP(nn.Module):
  def __init__(self, num_flow, feature_dim, feature_layer):
    super().__init__()
    self.num_flow = num_flow
    self.nout = 2
    self.net1 = AffineLayer(feature_dim, feature_layer, self.nout)
    self.net2 = AffineLayer(feature_dim, feature_layer, self.nout)
    self.net3 = AffineLayer(feature_dim, feature_layer, self.nout)
    self.net4 = AffineLayer(feature_dim, feature_layer, self.nout)


  def forward(self, x):
    batch_size = x.shape[0]
    x = x.float()
    x1, x2 = x[:, 0], x[:, 1]
    self.log_det = 0
    
    z1, z2, log_det = self.net1(x1, x2)
    self.log_det += log_det
    z1, z2, log_det = self.net2(z2, z1)
    self.log_det += log_det
    z1, z2, log_det = self.net3(z2, z1)
    self.log_det += log_det
    z1, z2, log_det = self.net4(z2, z1)
    self.log_det += log_det
    self.z = torch.stack((z1, z2), dim = -1)
    return self.log_det

  def predict(self, x):
    batch_size = x.shape[0]
    x = x.float()
    x1, x2 = x[:, 0], x[:, 1]
    z1, z2, _ = self.net1(x1, x2)
    z1, z2, _ = self.net2(z2, z1)
    z1, z2, _ = self.net3(z2, z1)
    z1, z2, _ = self.net4(z2, z1)
    return torch.stack((z1, z2), dim = -1)

  def log_prob(self, x):
    log_det = self(x)
    log_prob = torch.distributions.MultivariateNormal(torch.zeros(2).cuda(), torch.eye(2).cuda()).log_prob(self.z)
    return log_det + log_prob

  def loss(self, x):
    log_det = self(x)
    log_prob = torch.distributions.MultivariateNormal(torch.zeros(2).cuda(), torch.eye(2).cuda()).log_prob(self.z)
    log_prob = log_det.mean() + log_prob.mean()
    return -log_prob



class SimpleResnet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleResnet, self).__init__()
        feature_dim = 128
        self.net = [nn.Conv2d(in_dim, feature_dim, (3,3), stride=1, padding=1)]
        # self.net = [WeightedConv2d(in_dim, feature_dim, (3,3), stride=1, padding=1)]

        for _ in range(8):
            self.net.append(ResnetBlock(feature_dim))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(feature_dim, out_dim, (3,3), stride=1, padding=1))
        # self.net.append(WeightedConv2d(feature_dim, out_dim, (3,3), stride=1, padding=1))

        self.nets = nn.Sequential(*self.net)

    def forward(self, x):
        return self.nets(x)


class ActNorm(nn.Module):
    def __init__(self, channel, logscale_factor=1., scale=1.):
        super(ActNorm, self).__init__()
        self.initialized = False
        self.logscale_factor = logscale_factor
        self.scale = scale
        self.bias = nn.Parameter(torch.zeros(1, channel, 1)).cuda()
        self.logs = nn.Parameter(torch.zeros(1, channel, 1)).cuda()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)

        if not self.initialized:
            self.initialized = True
            unsqueeze = lambda x: x.unsqueeze(0).unsqueeze(-1).detach()
            b = -torch.sum(x, dim=(0, -1)) / (B * H * W)
            variance = unsqueeze(torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1))/(B * H * W))
            logs = torch.log(self.scale / (torch.sqrt(variance) + 1e-6)) / self.logscale_factor

            self.bias.data.copy_(unsqueeze(b).data)
            self.logs.data.copy_(logs.data)
            # First Minibatch has mean 0 and var 1

        output = x * torch.exp(self.logs)+ self.bias
        dlogdet = torch.sum(self.logs) * H * W # dim 1

        return output.view(B, C, H, W), dlogdet

    def reverse(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        output = (x - self.bias) * torch.exp(-self.logs)
        dlogdet = torch.sum(self.logs) * H * W # dim 1
        return output.view(B, C, H, W), -dlogdet


class CheckerboardCoupling(nn.Module):
    def __init__(self, channel, height, width, index):
        super(CheckerboardCoupling, self).__init__()

        self.mask = self.build_mask(height, width, index=index).cuda()
        self.res = SimpleResnet(channel, channel * 2)

    def build_mask(self, height, width, index=0):
        checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
        mask = torch.tensor(checkerboard).cuda()
        mask = mask.view(1, 1, height, width)
        if index:
          mask = 1 - mask
        return mask

    def forward(self, x):
        x_ = x * self.mask #[64, 3, 32, 32]
        log_s, t = torch.chunk(self.res(x_), 2, dim=1) #[64, 3, 32, 32], [64, 3, 32, 32]
        t = t * (1.0 - self.mask)
        # log_scale = log_s * (1.0 - self.mask)
        log_scale = torch.nn.Tanh()(log_s) * (1.0 - self.mask)
        z = x * torch.exp(log_scale) + t
        log_scale = log_scale.view(x.shape[0], -1).sum(-1)
        return z, log_scale
        # [64, 3, 32, 32], [64, 3, 32, 32]

    def reverse(self, x):
        mask = self.mask
        x_ = x * mask
        log_s, t = torch.chunk(self.res(x_), 2, dim=1)
        t = t * (1.0 - mask)
        log_scale = torch.nn.Tanh()(log_s) * (1.0 - mask)
        # log_scale = (log_s) * (1.0 - mask)
        z = (x - t) * torch.exp(-log_scale)
        return z, 0


class ChannelwiseCoupling(nn.Module):
    def __init__(self, channel, index):
        super(ChannelwiseCoupling, self).__init__()
        self.res = SimpleResnet(channel, channel * 2)
        self.mask = self.build_mask(channel, index)

    def build_mask(self, channel, index = 0):
        mask = [1 for _ in range(channel//2)] + [0 for _ in range(channel//2)]
        mask = torch.tensor(mask).cuda()
        mask = mask.view(1, -1, 1, 1)
        if index:
          mask = 1 - mask
        return mask

    def forward(self, x):
        x_ = x * self.mask #[64, 12, 16, 16], [1, 12, 1, 1] ->
        log_s, t = torch.chunk(self.res(x_), 2, dim=1) #[64, 3, 32, 32], [64, 3, 32, 32]
        t = t * (1.0 - self.mask)
        log_scale = torch.nn.Tanh()(log_s) * (1.0 - self.mask)
        z = x * torch.exp(log_scale) + t
        return z, log_scale.view(x.shape[0], -1).sum(-1) #[64, 12, 16, 16], [64, 12, 16, 16]

    def reverse(self, x):
        mask = self.mask
        x_ = x * mask
        log_s, t = torch.chunk(self.res(x_), 2, dim=1)
        t = t * (1.0 - mask)
        log_scale = torch.nn.Tanh()(log_s) * (1.0 - mask)
        z = (x - t) * torch.exp(-log_scale)
        return z, 0

class RealNVP(nn.Module):
    def __init__(self):
        super(RealNVP, self).__init__()
        self.prior = torch.distributions.Normal(torch.tensor(0.).cuda(), torch.tensor(1.).cuda())
        channel = 3
        height = width = 32

        nets1 = []
        flip = 1
        for _ in range(4):
            nets1.append(CheckerboardCoupling(channel, height, width, index = float(flip)))
            nets1.append(ActNorm(channel))
            flip = (flip==0)
        self.nets1 = nn.ModuleList(nets1)
        nets2 = []
        for _ in range(3):
            nets2.append(ChannelwiseCoupling(channel*4, float(flip)))
            nets2.append(ActNorm(channel*4))
            flip = (flip==0)
        self.nets2 = nn.ModuleList(nets2)

        nets3 = []
        for _ in range(3):
            nets3.append(CheckerboardCoupling(channel, height, width, index = float(flip)))
            nets3.append(ActNorm(channel))
            flip = (flip==0)
        self.nets3 = nn.ModuleList(nets3)


    def forward(self, x):
        x = x.float()
        self.log_det = torch.zeros((x.shape[0])).cuda()
        for layer in self.nets1:
          x, log_det = layer(x)
          self.log_det += log_det
        x = self.squeeze(x) # [64, 12, 16, 16]
        for layer in self.nets2:
          x, log_det = layer(x)
          self.log_det += log_det
        x = self.unsqueeze(x)
        for layer in self.nets3:
          x, log_det = layer(x)
          self.log_det += log_det
        return x, self.log_det

    def loss(self, x):
        z, log_diag_J = self(x) #[64]
        B, C, H, W = z.shape
        log_diag_J = log_diag_J.mean()/(C * H * W)
        log_prior_prob = self.prior.log_prob(z).mean()
        log_p = log_prior_prob + log_diag_J
        loss = -log_p + np.log(4)
        return loss

    def squeeze(self, x):
        # [b, c, h, w] --> [b, c*4, h//2, w//2]
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, C*4, H//2, W//2)
        return x

    def unsqueeze(self, x):
        # [b, c*4, h//2, w//2] --> [b, c, h, w]
        B, C, H, W = x.shape
        x = x.reshape(B, C//4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(B, C//4, H*2, W*2)
        return x

    def reverse(self, z):
        x = z.float()
        for layer in reversed(self.nets3):
          x, _ = layer.reverse(x)
        x = self.squeeze(x)
        for layer in reversed(self.nets2):
          x, _ = layer.reverse(x)
        x = self.unsqueeze(x)
        for layer in reversed(self.nets1):
          x, _ = layer.reverse(x)
        return x


    def interpol(self, samples1, samples2):
        latents1, _ = self(samples1)
        latents2, _ = self(samples2)
        to_invert = torch.zeros(samples1.shape[0], 6, *samples1.shape[1:])
        alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for a in range(len(alphas)):
          interp = alphas[a]
          x = self.reverse(interp * latents1 + (1-interp) * latents2)
          to_invert[:,a] = x
        return to_invert.view(len(alphas) * len(samples1), *samples1.shape[1:]).detach().cpu().numpy()
