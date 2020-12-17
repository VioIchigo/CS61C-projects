from deepul.hw2_helper import *
import deepul.pytorch_util as ptu
from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.utils.data as data
import math

class Flow(nn.Module):
  def __init__(self, num_mix, hidden_sizes):
    "does a 2 dim AR flow"
    super().__init__()
    self.input_shape = 1 #autoregressive, so doesn't feed in one dimension
    self.num_mix = num_mix
    self.nout = num_mix * 3
    self.hidden_sizes = hidden_sizes

    # define a simple MLP neural net
    self.net = []
    hs = [1] + self.hidden_sizes + [self.nout]
    for h0, h1 in zip(hs, hs[1:]):
      self.net.extend([
        nn.Linear(h0, h1),
        nn.Sigmoid(),
      ])
    self.net.pop() 

    self.net1 = nn.Sequential(nn.Linear(1, self.nout)) #use the biases as the logits
    self.net2 = nn.Sequential(*self.net)

  def forward(self, x):
    #gives the logits in B,3,num_mix,2

    batch_size = x.shape[0]
    x = x.float()
    # x = x.view(batch_size, self.nin)

    x1 = x[:, 0]
    # x2 = x[:, 1]
    
    dummies = torch.zeros((batch_size,1)).float().cuda()
    x1_logits = self.net1(dummies)
    x2_logits = self.net2(x1.unsqueeze(-1))
    # logits = self.net(x).view(batch_size, self.nin, self.d)

    logits = torch.stack([x1_logits.view(batch_size, 3, self.num_mix), x2_logits.view(batch_size, 3, self.num_mix)], -1) #B, 15, 2
    return logits
    # return logits.permute(0, 2, 1).contiguous().view(batch_size, self.d, *self.input_shape)

  def log_prob(self, x):
    pred = self(x)
    # pred = pred.view((pred.shape[0], 3, 5, pred.shape[-1]))
    weight1 = nn.Softmax(1)(pred[:,0,:,0]) #shape [B, 5]
    weight2 = nn.Softmax(1)(pred[:,0,:,1]) #shape [B, 5]

    mean = pred[:, 1] #shape [B, 5, 2]
    log_sig = pred[:, 2] #shape [B, 5, 2]

    cdf1 = torch.zeros((x.shape[0])).cuda()
    cdf2 = torch.zeros((x.shape[0])).cuda()
    pdf1 = torch.zeros((x.shape[0])).cuda()
    pdf2 = torch.zeros((x.shape[0])).cuda()
    for i in range(5):
      cdf1 += weight1[:, i] * self.get_cdf(x[:, 0], mean[:, i, 0], log_sig[:, i, 0])
      cdf2 += weight2[:, i] * self.get_cdf(x[:, 1], mean[:, i, 1], log_sig[:, i, 1])
      pdf1 += weight1[:, i] * torch.exp(self.get_log_pdf(x[:, 0], mean[:, i, 0], log_sig[:, i, 0]))
      pdf2 += weight2[:, i] * torch.exp(self.get_log_pdf(x[:, 1], mean[:, i, 1], log_sig[:, i, 1]))
    return torch.stack((cdf1, cdf2), dim = -1), (torch.log(pdf1 + 1e-12) + torch.log(pdf2 + 1e-12))

  def get_cdf(self, value, means, log_sigs):
    sig_recip = torch.exp(-log_sigs)
    return 0.5 * (1 + torch.erf((value - means) * sig_recip / math.sqrt(2)))

  def get_log_pdf(self, value, means, log_sigs):
    var = torch.exp(log_sigs)**2
    return -((value - means) ** 2) / (2 * var) - log_sigs - math.log(math.sqrt(2 * math.pi))

  def loss(self, x):
    cdf, pdf = self.log_prob(x)
    loss = - pdf.mean() 
     #- torch.log(cdf).mean()
    return loss


import torch.nn.functional as F

class MaskConv2d(nn.Conv2d):
  def __init__(self, mask_type, *args, conditional_size=None,
               color_conditioning=False, **kwargs):
    assert mask_type == 'A' or mask_type == 'B'
    super().__init__(*args, **kwargs)
    self.conditional_size = conditional_size
    self.color_conditioning = color_conditioning
    self.register_buffer('mask', torch.zeros_like(self.weight))
    self.create_mask(mask_type)
    if self.conditional_size:
      if len(self.conditional_size) == 1:
        self.cond_op = nn.Linear(conditional_size[0], self.out_channels)
      else:
        self.cond_op = nn.Conv2d(conditional_size[0], self.out_channels,
                                 kernel_size=3, padding=1)

  def forward(self, input, cond=None):
    batch_size = input.shape[0]
    out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                   self.padding, self.dilation, self.groups)
    if self.conditional_size:
      if len(self.conditional_size) == 1:
        out = out + self.cond_op(cond).view(batch_size, -1, 1, 1)
      else:
        out = out + self.cond_op(cond)
    return out

  def create_mask(self, mask_type):
    k = self.kernel_size[0]
    self.mask[:, :, :k // 2] = 1
    self.mask[:, :, k // 2, :k // 2] = 1
    if self.color_conditioning:
      assert self.in_channels % 3 == 0 and self.out_channels % 3 == 0
      one_third_in, one_third_out = self.in_channels // 3, self.out_channels // 3
      if mask_type == 'B':
        self.mask[:one_third_out, :one_third_in, k // 2, k // 2] = 1
        self.mask[one_third_out:2*one_third_out, :2*one_third_in, k // 2, k // 2] = 1
        self.mask[2*one_third_out:, :, k // 2, k // 2] = 1
      else:
        self.mask[one_third_out:2*one_third_out, :one_third_in, k // 2, k // 2] = 1
        self.mask[2*one_third_out:, :2*one_third_in, k // 2, k // 2] = 1
    else:
      if mask_type == 'B':
        self.mask[:, :, k // 2, k // 2] = 1

class ResBlock(nn.Module):
  def __init__(self, in_channels, **kwargs):
    super().__init__()
    self.block = nn.ModuleList([
        nn.Tanh(),
        MaskConv2d('B', in_channels, in_channels // 2, 1, **kwargs),
        nn.Tanh(),
        MaskConv2d('B', in_channels // 2, in_channels // 2, 7, padding=3, **kwargs),
        nn.Tanh(),
        MaskConv2d('B', in_channels // 2, in_channels, 1, **kwargs)
    ])

  def forward(self, x, cond=None):
    out = x
    for layer in self.block:
      if isinstance(layer, MaskConv2d):
        out = layer(out, cond=cond)
      else:
        out = layer(out)
    return out + x

class LayerNorm(nn.LayerNorm):
  def __init__(self, color_conditioning, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.color_conditioning = color_conditioning

  def forward(self, x):
    x = x.permute(0, 2, 3, 1).contiguous()
    x_shape = x.shape
    if self.color_conditioning:
      x = x.contiguous().view(*(x_shape[:-1] + (3, -1)))
    x = super().forward(x)
    if self.color_conditioning:
      x = x.view(*x_shape)
    return x.permute(0, 3, 1, 2).contiguous()

class PixelCNN(nn.Module):
  def __init__(self, input_shape, n_colors, n_filters=64,
               kernel_size=7, n_layers=5,
               conditional_size=None, use_resblock=False,
               color_conditioning=False, gaussian_num = 5):
    super().__init__()
    assert n_layers >= 2
    n_channels = input_shape[0]
    self.gaussian_num = gaussian_num

    kwargs = dict(conditional_size=conditional_size,
                  color_conditioning=color_conditioning)
    if use_resblock:
      block_init = lambda: ResBlock(n_filters, **kwargs)
    else:
      block_init = lambda: MaskConv2d('B', n_filters, n_filters,
                                      kernel_size=kernel_size,
                                      padding=kernel_size // 2, **kwargs)

    model = nn.ModuleList([MaskConv2d('A', n_channels, n_filters,
                                      kernel_size=kernel_size,
                                      padding=kernel_size // 2, **kwargs)])
    for _ in range(n_layers):
      if color_conditioning:
        model.append(LayerNorm(color_conditioning, n_filters // 3))
      else:
        model.append(LayerNorm(color_conditioning, n_filters))
      model.extend([nn.Tanh(), block_init()])
    model.extend([nn.Tanh(), MaskConv2d('B', n_filters, n_filters, 1, **kwargs)])
    model.extend([nn.Tanh(), MaskConv2d('B', n_filters, n_colors * n_channels, 1, **kwargs)])

    if conditional_size:
      if len(conditional_size) == 1:
        self.cond_op = lambda x: x # No preprocessing conditional if one hot
      else:
        self.cond_op = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Tanh()
        )

    self.net = model
    self.input_shape = input_shape
    self.n_colors = n_colors
    self.n_channels = n_channels
    self.color_conditioning = color_conditioning
    self.conditional_size = conditional_size
    self.gaussians = [0 for i in range(5)]
    self.weights = [0 for i in range(5)]

  def forward(self, x, cond=None):
    batch_size = x.shape[0]
    out = (x.float() / (self.n_colors - 1) - 0.5) / 0.5
    if self.conditional_size:
      cond = self.cond_op(cond)
    for layer in self.net:
      if isinstance(layer, MaskConv2d) or isinstance(layer, ResBlock):
        out = layer(out, cond=cond)
      else:
        out = layer(out)

    if self.color_conditioning:
      return out.view(batch_size, self.n_channels, self.n_colors,
                      *self.input_shape[1:]).permute(0, 2, 1, 3, 4)
    else:
      return out.view(batch_size, self.n_colors, *self.input_shape)

  def log_prob(self, x):
    pred = self(x) #torch.Size([128, 15, 1, 20, 20])
    batch_size, _, _, d2, d3 = pred.shape
    pred = pred.reshape((batch_size, 3, self.gaussian_num, d2, d3)) #torch.Size([128, 3, 5, 1, 20, 20])
    pdf = torch.zeros((batch_size, d2, d3)).cuda()
    weight = nn.Softmax(1)(pred[:, 0]) #[128, 5, 20, 20]
    mean = pred[:, 1] #[128, 5, 20, 20]
    log_var = pred[:, 2] #[128, 5, 20, 20]
    for i in range(self.gaussian_num):
      pdf += weight[:, i] * torch.exp(Normal(mean[:, i], torch.exp(log_var[:, i])).log_prob(x[:, 0]))
    return pdf

  def icdf(self, x, z):
    pred = self(x)
    batch_size, _, _, d2, d3 = pred.shape
    pred = pred.reshape((batch_size, 3, self.gaussian_num, d2, d3))
    icdf = torch.zeros((batch_size, d2, d3)).cuda()
    weight = nn.Softmax(1)(pred[:, 0]) #[128, 5, 20, 20]
    mean = pred[:, 1] #[128, 5, 20, 20]
    log_var = pred[:, 2] #[128, 5, 20, 20]
    for i in range(self.gaussian_num):
      icdf += weight[:, i] * torch.exp(Normal(mean[:, i], torch.exp(log_var[:, i])).icdf(z[:, 0])) 
    icdf = (icdf - icdf.min())/(icdf.max()-icdf.min())
    return icdf


  def loss(self, x, cond=None):
    pdf = self.log_prob(x)
    return -torch.log(pdf + 1e-8).mean() + np.log(2)

  def sample(self, n, cond=None):
    samples = torch.zeros(n, *self.input_shape).cuda()
    zs = torch.rand(n, *self.input_shape).cuda()
    with torch.no_grad():
      for r in range(self.input_shape[1]):
        for c in range(self.input_shape[2]):
          for k in range(self.n_channels):
              icdf = self.icdf(samples, zs)[:, r, c]
              samples[:, k, r, c] = icdf
    return samples.permute(0, 2, 3, 1).cpu().numpy()

class TransformModule(torch.distributions.Transform, torch.nn.Module):
    """
    Transforms with learnable parameters such as normalizing flows should inherit from this class rather
    than `Transform` so they are also a subclass of `nn.Module` and inherit all the useful methods of that class.
    """

    def __init__(self):
        super(TransformModule, self).__init__()

    def __hash__(self):
        return super(torch.nn.Module, self).__hash__()

def process_data(data):
  data = data.astype(float)
  rand = np.random.uniform(0, 1, data.shape)
  data += rand
  data = data/2
  return data