
from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.utils.data as data
import math


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
        # Broadcast across height and width of image and add as conditional bias
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
        nn.ReLU(),
        MaskConv2d('B', in_channels, in_channels // 2, 1, **kwargs),
        nn.ReLU(),
        MaskConv2d('B', in_channels // 2, in_channels // 2, 7, padding=3, **kwargs),
        nn.ReLU(),
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
  def __init__(self, input_shape=(8,8), n_colors=128, n_filters=64,
               kernel_size=7, n_layers=10, 
               conditional_size=None, use_resblock=True,
               color_conditioning=False):
    super().__init__()
    self.z_emb = Codebook(128, 64) #codebook with size 

    assert n_layers >= 2
    n_channels = 64

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
      model.extend([nn.ReLU(), block_init()])

    model.extend([nn.ReLU(), MaskConv2d('B', n_filters, 512, 1, **kwargs)])
    model.extend([nn.ReLU(), MaskConv2d('B', 512, n_colors, 1, **kwargs)])


    if conditional_size:
      if len(conditional_size) == 1:
        self.cond_op = lambda x: x # No preprocessing conditional if one hot
      else:
        # For Grayscale PixelCNN (some preprocessing on the binary image)
        self.cond_op = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )

    self.net = model
    self.input_shape = input_shape
    self.n_colors = n_colors
    self.n_channels = n_channels
    self.color_conditioning = color_conditioning
    self.conditional_size = conditional_size

  def forward(self, x, cond=None): 
    batch_size = x.shape[0]
    x = x.long()
    embed = self.z_emb.embedding(x) 
    out = embed.permute(0, 3, 1, 2).contiguous()
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

  def loss(self, x, cond=None):
    x = x.long() 
    return F.cross_entropy(self(x, cond=cond), x)

  def sample(self, n, cond=None):
    samples = torch.zeros(n, *self.input_shape).cuda()
    with torch.no_grad():
      for r in range(self.input_shape[0]):
        for c in range(self.input_shape[1]):
          logits = self(samples, cond=cond)
          logits = logits[:, :, r, c]
          probs = F.softmax(logits, dim=1)
          samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)
    return samples


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(dim),
                                 nn.ReLU(),
                                 nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                                 nn.ReLU(),
                                 nn.Conv2d(dim, dim, 1, stride=1, padding=0))

    def forward(self, x):
        return x + self.net(x)

class Encoder(nn.Module):
    def __init__(self, channel_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channel_dim, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 4, 2, 1),
            ResnetBlock(256),
            ResnetBlock(256),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channel_dim):
        super().__init__()
        self.model = nn.Sequential(
            ResnetBlock(256),
            ResnetBlock(256),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 3, 4, 2, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Codebook(nn.Module):
  def __init__(self, K, D):
    super().__init__()
    self.embedding = nn.Embedding(K, D)
    self.embedding.weight.data.uniform_(-1/K, 1/K)
    self.K = K
    self.D = D
    self.beta = 1.

  def forward(self, x, B, H, W):
    distances = (torch.sum(x**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(x, self.embedding.weight.t()))
    encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
    encodings = torch.zeros(encoding_indices.shape[0], self.K).cuda()
    encodings.scatter_(1, encoding_indices, 1)
    quantized = torch.matmul(encodings, self.embedding.weight).view(x.shape)

    e_latent_loss = F.mse_loss(quantized.detach(), x)
    q_latent_loss = F.mse_loss(x.detach(), quantized)
    loss = q_latent_loss + self.beta * e_latent_loss

    quantized = x + (quantized - x).detach()
    avg_probs = torch.mean(encodings, dim=0)
    quantized = quantized.view(B, H, W, self.D)
    quantized = quantized.permute(0, 3, 1, 2).contiguous()

    return loss, quantized, encoding_indices


class VQ_VAE(nn.Module):
  def __init__(self, K, D, channel_dim=3):
    super().__init__()
    self.encoder = Encoder(channel_dim)
    self.decoder = Decoder(channel_dim)
    self.codebook = Codebook(K, D)
  
  def to_latent(self, x):
    z = self.encoder(x) #[128=B, 256=D, 8=H, 8=W]
    B, D, H, W = z.shape
    z = z.permute(0, 2, 3, 1).contiguous()
    z = z.view(B * H * W, D) #[BxHxW, D]
    loss, quantized, encoding_indices = self.codebook(z, B, H, W)
    return loss, quantized, encoding_indices

  def loss(self, x):
    x = x.float()
    loss, quantized, _ = self.to_latent(x)
    recon = self.decoder(quantized)
    recon_loss = F.mse_loss(recon, x)
    loss = recon_loss + loss
    return loss

  def reconstruct(self, x):
    x = x.float()
    loss, quantized, _ = self.to_latent(x)
    recon = self.decoder(quantized)
    recon = torch.clamp(recon, -1, 1)
    return recon.detach().cpu().numpy()
