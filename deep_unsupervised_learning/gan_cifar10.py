from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.utils.data as data
import math

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1).contiguous()
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().reshape(batch_size, s_height, s_width,
                                                                                      s_depth)
        output = output.permute(0, 3, 1, 2).contiguous()
        return output


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1).contiguous()
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.permute(0, 3, 1, 2).contiguous()
        return output

# Spatial Upsampling with Nearest Neighbors
class Upsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.main = nn.Sequential(
            DepthToSpace(block_size=2),
            nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding),
        )

    def forward(self, x):
        x = torch.cat([x, x, x, x], dim=1)
        x = self.main(x)
        return x
 
 
# Spatial Downsampling with Spatial Mean Pooling
class Downsample_Conv2d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=(3, 3), stride=1, padding=1):
        super().__init__()
        self.ly1 = SpaceToDepth(2)
        self.ly2 = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.ly1(x)
        B, D, _, _ = x.shape
        x = x.view(B, 4, D//4, *x.shape[2:])
        x = torch.sum(x, dim = 1) / 4.0 #[128, 128, 16, 16]
        x = self.ly2(x)
        return x


class ResnetBlockUp(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm2d(in_dim), #[128, 256, 4, 4]
            nn.ReLU(), 
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1), #[128, 128, 4, 4]
            nn.ReLU(),
            Upsample_Conv2d(n_filters, n_filters, kernel_size, padding=1),
        )

        self.short = Upsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)


    def forward(self, x):
        residual = self.main(x)
        shortcut = self.short(x)
        return residual + shortcut #[128, 128, 8, 8]


class ResnetBlockDown(nn.Module):
    def __init__(self, in_dim, kernel_size=(3, 3), n_filters=256):
        super().__init__()

        self.main = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, n_filters, kernel_size, padding=1),
            nn.ReLU(),
            Downsample_Conv2d(n_filters, n_filters, kernel_size, padding=1),
        )


        self.short = Downsample_Conv2d(in_dim, n_filters, kernel_size=(1, 1), padding=0)


    def forward(self, x): #[128, 128, 32, 32]
        residual = self.main(x) #[128, 128, 16, 16]
        shortcut = self.short(x)
        return residual + shortcut
          
class G(nn.Module):
    def __init__(self, n_filters = 128):
        super().__init__()
        self.linear = nn.Linear(128, 4*4*256)
        self.net = nn.Sequential(
            ResnetBlockUp(in_dim=256, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockUp(in_dim=n_filters, n_filters=n_filters),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 3, kernel_size=(3, 3), padding=1),
            nn.Tanh(),
            )
        self.dist = torch.distributions.Normal(0, 1)

    def forward(self, n_samples=1024): #sampling
        z = self.dist.sample([n_samples, 128]).cuda()
        bs = n_samples
        z = self.linear(z)
        z = z.view(bs, 256, 4, 4)
        samples = self.net(z)
        return samples
        
class D(nn.Module):
    def __init__(self, n_filters = 128):
        super().__init__()
        self.linear = nn.Linear(4*4*256, 128)
        self.linear2 = nn.Linear(128, 1)
        self.net = nn.Sequential(
            nn.Conv2d(3, n_filters, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockDown(in_dim=n_filters, n_filters=n_filters),
            ResnetBlockDown(in_dim=n_filters, n_filters=256),
            )

    def forward(self, x):
        x = x.float()
        bs = x.shape[0]
        x = self.net(x) #[128, 256, 4, 4]
        x = x.view(bs, -1)
        x = self.linear(x) #[128, 128]

        x = nn.ReLU()(x)
        x = self.linear2(x)
        return x


class GAN(nn.Module):
    def __init__(self, D, G, n_critic):
        super(GAN, self).__init__()
        self.D = D
        self.G = G
        self.n_critic = n_critic
        self.lamb = 10


    def D_loss(self, x):
        x = x.float()
        bs = x.shape[0]
        fake = self.G(bs)

        fake_d = self.D(fake)
        gradient_penalty = self.gradient_penalty(x, fake)

        return fake_d.mean() - self.D(x).mean() + self.lamb * gradient_penalty

    def G_loss(self, x):
        x = x.float()
        bs = x.shape[0]
        fake = self.D(self.G(bs))
        return -torch.mean(fake)

    def gradient_penalty(self, real, fake):
        bs = real.shape[0]
        epis = torch.from_numpy(np.random.uniform(0,1,bs)).view(bs, 1, 1, 1).cuda()
        interpolates = epis * real + ((1 - epis) * fake)
        d_interpolates = self.D(interpolates)

        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(d_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


def train(model, train_loader, D_optimizer, G_optimizer, epoch):
  model.train()
  
  d_losses = []
  count = 1
  for x in train_loader:
    x = x.cuda().contiguous()

    loss1 = model.D_loss(x)

    D_optimizer.zero_grad()
    loss1.backward()
    D_optimizer.step()

    if count % model.n_critic == 0:
        loss2 = model.G_loss(x)
        G_optimizer.zero_grad()
        loss2.backward()
        G_optimizer.step()

    d_losses.append(loss1.item())
    count += 1
  return d_losses


def preprocess(data):
    data *= 2 #[0, 2]
    data -= 1 #[-1, 1]
    return data

def normalize(data):
    data += 1 #[0, 2]
    data /= 2 #[0, 1]
    data = np.clip(data, 0, 1)
    return data

def train_epochs(model, train_loader, train_args):
    epochs, lr, betas = train_args['epochs'], train_args['lr'], train_args['betas']
    

    d_losses = []
    # interval = epochs // lr.shape[0]
    # print(interval)
    D_optimizer = optim.Adam(model.D.parameters(), lr=2e-4, betas=betas)
    G_optimizer = optim.Adam(model.G.parameters(), lr=2e-4, betas=betas)
    for epoch in range(epochs):
        # if epoch % interval  == 0:
        #     idx = (epoch // interval)
        #     D_optimizer = optim.Adam(model.D.parameters(), lr=lr[idx], betas=betas)
        #     G_optimizer = optim.Adam(model.G.parameters(), lr=lr[idx], betas=betas)
        loss1 = train(model, train_loader, D_optimizer, G_optimizer, epoch)
        d_losses.extend(loss1)

        samples = model.G(1000).detach().cpu().numpy()
        samples = np.transpose(samples, (0, 2, 3, 1))
        samples = normalize(samples)
        print("Inception score:", calculate_is(samples.transpose([0, 3, 1, 2])))
        plot_gan_training(d_losses, 'Q2 Losses', 'results/q2_losses.png')
        show_samples(samples[:100] * 255.0, fname='results/q2_samples.png', title=f'CIFAR-10 generated samples')

    return d_losses



def q2(train_data):
    """
    train_data: An (n_train, 3, 32, 32) numpy array of CIFAR-10 images with values in [0, 1]

    Returns
    - a (# of training iterations,) numpy array of WGAN critic train losses evaluated every minibatch
    - a (1000, 32, 32, 3) numpy array of samples from your model in [0, 1]. 
        The first 100 will be displayed, and the rest will be used to calculate the Inception score. 
    """

    """ YOUR CODE HERE """
    _, C, H, W = train_data.shape
    train_data = preprocess(train_data)
    betas=(0, 0.9)
    epochs = 100
    lr = np.linspace(2e-4, 0, num=4)
    n_critic = 5
    model = GAN(D(), G(), n_critic).cuda()
    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
    d_losses = train_epochs(model, train_loader, dict(epochs=epochs, lr=lr, betas=betas))
    samples = model.G(1000).detach().cpu().numpy()
    samples = np.transpose(samples, (0, 2, 3, 1))
    return d_losses, samples