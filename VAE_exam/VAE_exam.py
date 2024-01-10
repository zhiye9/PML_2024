from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions.continuous_bernoulli import ContinuousBernoulli
import os
os.chdir('/home/zhi/data/MLSP/VAE_exam')

torch.manual_seed(1)
device = torch.device("cpu")
# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=128, shuffle=False)

def cont_bern_log_norm(lam, l_lim=0.49, u_lim=0.51):
    cut_lam = torch.where(torch.logical_or(torch.less(lam, l_lim), torch.greater(lam, u_lim)), lam, l_lim * torch.ones_like(lam))
    if (torch.log(torch.abs(2.0 * torch.atanh(1 - 2.0 * cut_lam))) == torch.log(torch.abs(1 - 2.0 * cut_lam)) ).all():
        print("1 inf")
    if (torch.log(torch.abs(1 - 2.0 * cut_lam)) == 0).all():
        print("2 inf")
    log_norm = torch.log(torch.abs(2.0 * torch.atanh(1 - 2.0 * cut_lam))) - torch.log(torch.abs(1 - 2.0 * cut_lam))
    taylor = torch.log(torch.tensor(2.0)) + torch.tensor(4.0 / 3.0 )* (lam - 0.5).pow(2) + torch.tensor(104.0 / 45.0) * (lam - 0.5).pow(4)
    logc = torch.where(torch.logical_or(torch.less(lam, l_lim), torch.greater(lam, u_lim)), log_norm, taylor)
    return logc        

def conti_berno(y, x):
    # x: input, z: reconstruct variable
    # print('constant', cont_bern_log_norm(y)[0][0])
    BCE_loss = torch.mean(torch.sum(x*torch.log(y) + (1-x)*torch.log(1-y) + cont_bern_log_norm(y), axis = 0))
    #BCE = F.binary_cross_entropy(y, x.view(-1, 784), reduction='sum')
    #logC = cont_bern_log_norm(y).sum()
    return -BCE_loss

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 300)
        self.fc21 = nn.Linear(300, 2)
        self.fc22 = nn.Linear(300, 2)
        self.fc3 = nn.Linear(2, 300)
        self.fc4 = nn.Linear(300, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE_berno = conti_berno(recon_x, x.view(-1, 784))
    print(BCE_berno)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return KLD + BCE_berno


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(128, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, 10 + 1):
        ry = train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 2).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

# def plot_latent(model, data, num_batches=128):
#     for i, (x, y) in enumerate(data):
#         mu, logvar = model.encode(x.view(-1, 784))
#         # z = z[0]+eps*z[1].detach().numpy()
#         z = model.reparameterize(mu, logvar).detach().numpy()
#         plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
#         if i > num_batches:
#             plt.colorbar()
#             break
# plt.figure(dpi=400)
# plot_latent(model, test_loader)

# def plot_reconstructed(model, r0=(-2.5, 2.5), r1=(-2.5, 2.5), n=15):
#     w = 28
#     img = np.zeros((n*w, n*w))
#     for i, y in enumerate(np.linspace(*r1, n)):
#         for j, x in enumerate(np.linspace(*r0, n)):
#             z = torch.Tensor([[x, y]]).to(device)
#             x_hat = model.decode(z)
#             x_hat = x_hat.reshape(28, 28).detach().numpy()
#             img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
#     plt.xlabel('z1')
#     plt.ylabel('z2')
#     plt.imshow(img, extent=[*r0, *r1])
# plt.figure(dpi=400)
# plot_reconstructed(model)