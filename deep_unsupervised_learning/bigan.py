from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import torch.utils.data as data
import math


class E(nn.Module):
    def __init__(self, out_dim=50):
        """ x -> E(x) """
        super().__init__()
        self.out_dim = out_dim
        self.img_dim = 28 * 28

        self.main = nn.Sequential(
            # x: [B, 1, 28, 28]
            nn.Linear(self.img_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x):
        return self.main(x)


          
class G(nn.Module):
    def __init__(self, z_dim = 50):
        super().__init__()
        """ z -> G(z) """
        self.z_dim = z_dim
        self.img_dim = 28 * 28 

        self.main = nn.Sequential(
            nn.Linear(self.z_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.img_dim),
            nn.Tanh()
        )
        self.dist = torch.distributions.Normal(0, 1)


    def forward(self, n):
        samples = self.dist.sample([n, self.z_dim]).cuda()
        g_samples = self.main(samples)
        return samples, g_samples

    def recon(self, x):
        x = x.float()
        return self.main(x)


        
class D(nn.Module):
    def __init__(self):
        super().__init__()

        self.img_dim = 28 * 28 
        self.z_dim = 50 

        self.in_dim = self.img_dim + self.z_dim

        self.main = nn.Sequential(
            nn.Linear(self.in_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        return self.main(x)

class L(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim = 50
        out = 10
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, out, bias = False),
            nn.Softmax(),
        )
        self.criterion = nn.CrossEntropyLoss()

    def loss(self, x, label):
        x = x.float()
        pred = self.linear(x) #[B, C]
        loss = self.criterion(pred, label.long())
        return loss


class BiGAN(nn.Module):
    def __init__(self, D, G, E):
        super(BiGAN, self).__init__()
        self.D = D
        self.G = G
        self.E = E

    def D_loss(self, x):
        x = x.float()
        bs = x.shape[0]
        z, g_z = self.G(bs) #50, img_dim

        e_x = self.E(x) #50

        zs = torch.cat([g_z, z], dim = -1)
        xs = torch.cat([x, e_x], dim = -1)
        d_zs = self.D(zs)
        d_xs = self.D(xs)

        loss1 = F.binary_cross_entropy(d_zs, torch.zeros(bs).cuda())
        loss2 = F.binary_cross_entropy(d_xs, torch.ones(bs).cuda())
        return loss1 + loss2


    def G_loss(self, x):
        return -self.D_loss(x)


def train_encoder(encoder, classifier, optimizer, train_loader):
    for data in train_loader:
        x = data[0]
        label = data[1]
        x = x.cuda().contiguous()
        bs = x.shape[0]
        x = x.view(bs, -1)
        label = label.cuda().contiguous()
        loss = classifier.loss(encoder(x), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_encoder(encoder, classifier, optimizer, test_loader):
    losses = []
    for x in test_loader:
        label = x[1]
        x = x[0]
        x = x.cuda().contiguous()
        bs = x.shape[0]
        x = x.view(bs, -1)
        label = label.cuda().contiguous()
        loss = classifier.loss(encoder(x), label)
        losses.append(loss.detach().cpu().numpy())
    return losses

def train(model, train_loader, D_optimizer, G_optimizer, epoch):
  model.train()
  
  d_losses = []
  for data_pair in train_loader:
    x = data_pair[0]
    label = data_pair[1]

    x = x.cuda().contiguous()
    bs = x.shape[0]
    x = x.view(bs, -1)
    loss1 = model.D_loss(x)

    D_optimizer.zero_grad()
    loss1.backward()
    D_optimizer.step()

    loss2 = model.G_loss(x)
    G_optimizer[0].zero_grad()
    G_optimizer[1].zero_grad()
    loss2.backward()
    G_optimizer[0].step()
    G_optimizer[1].step()

    d_losses.append(loss1.item())
    
  return d_losses

def normalize(data):
    return (data-data.min())/(data.max()-data.min())

def normalize255(data):
    return 255.*(data-data.min())/(data.max()-data.min())

def train_epochs(model, train_loader, test_loader, train_args, sample_data):
    epochs, lr, betas = train_args['epochs'], train_args['lr'], train_args['betas']
    D_optimizer = optim.Adam(model.D.parameters(), lr=lr, betas=betas, weight_decay = 2.5e-5)
    G_optimizer = [optim.Adam(model.G.parameters(), lr=lr, betas=betas, weight_decay = 2.5e-5), optim.Adam(model.E.parameters(), lr=lr, betas=betas, weight_decay = 2.5e-5)]

    d_losses = []
    for epoch in range(epochs):
        print("epoch: ", epoch)
        loss1 = train(model, train_loader, D_optimizer, G_optimizer, epoch)
        d_losses.extend(loss1)

        real = sample_data.reshape((20, -1)).cuda().float()
        recon = model.G.recon(model.E(real).cuda())*255.
        real = real.detach().cpu().numpy()
        recon = recon.detach().cpu().numpy()
        real = real.reshape((20, 28, 28, 1))
        recon = recon.reshape((20, 28, 28, 1))
        real = normalize255(real)
        recon = normalize255(recon)
        reconstructions = np.concatenate((real, recon), axis = 0)

        samples = model.G(100)[1].detach().cpu().numpy().reshape((100, 28, 28, 1))


        plot_gan_training(d_losses, 'Q3 Losses', 'results/q3_gan_losses.png')
        plot_q3_supervised(classifier1_loss, classifier2_loss, 'Linear classification losses', 'results/q3_supervised_losses.png')
        samples = normalize(samples)
        show_samples(samples * 255.0, fname='results/q3_samples.png', title='BiGAN generated samples')
        # reconstructions = normalize(reconstructions)
        show_samples(reconstructions, nrow=20, fname='results/q3_reconstructions.png', title=f'BiGAN reconstructions')
    random_encoder = E().cuda()
    E_optimizer = optim.SGD(L1.parameters(), lr=1e-3)
    rand_e_optimizer = optim.SGD(L2.parameters(), lr=1e-3)
    L1 = L().cuda()
    L2 = L().cuda()
    classifier1_loss = []
    classifier2_loss = []
    for epoch in range(20):
        train_encoder(model.E, L1, E_optimizer, train_loader)
        train_encoder(random_encoder, L2, rand_e_optimizer, train_loader)
    classifier1_loss.extend(test_encoder(model.E, L1, E_optimizer, test_loader))
    classifier2_loss.extend(test_encoder(random_encoder, L2, rand_e_optimizer, test_loader))
    print('BiGAN final linear classification loss:', classifier1_loss[-1])
    print('Random encoder linear classification loss:', classifier2_loss[-1])

    return d_losses




def q3(train_data, test_data):
    """
    train_data: A PyTorch dataset that contains (n_train, 28, 28) MNIST digits, normalized to [-1, 1]
                Documentation can be found at torchvision.datasets.MNIST, and it may be easiest to directly create a DataLoader from this variable
    test_data: A PyTorch dataset that contains (n_test, 28, 28) MNIST digits, normalized to [-1, 1]
                Documentation can be found at torchvision.datasets.MNIST

    Returns
    - a (# of training iterations,) numpy array of BiGAN minimax losses evaluated every minibatch
    - a (100, 28, 28, 1) numpy array of BiGAN samples that lie in [0, 1]
    - a (40, 28, 28, 1) numpy array of 20 real image / reconstruction pairs
    - a (# of training epochs,) numpy array of supervised cross-entropy losses on the BiGAN encoder evaluated every epoch 
    - a (# of training epochs,) numpy array of supervised cross-entropy losses on a random encoder evaluated every epoch 
    """

    """ YOUR CODE HERE """

    betas=(0.5, 0.999)
    lr = 2e-4
    epochs = 1
    model = BiGAN(D(), G(), E()).cuda()
    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=128, shuffle=True)
    d_losses = train_epochs(model, train_loader, test_loader, dict(epochs=epochs, lr=lr, betas=betas), sample_data = train_data.data[:20])
    return 