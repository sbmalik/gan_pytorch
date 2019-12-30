import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import dataloader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
import os
import argparse

from utils import *

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=12, help="no of epochs")
parser.add_argument("--batch_size", type=int, default=64, help="no of epochs")
parser.add_argument("--lrelu_a", type=float, default=0.01, help="no of epochs")
parser.add_argument("--lr", type=float, default=0.0002, help="no of epochs")
parser.add_argument("--b1", type=float, default=0.5, help="no of epochs")
parser.add_argument("--b2", type=float, default=0.999, help="no of epochs")
parser.add_argument("--latent_dim", type=int, default=100, help="no of epochs")
parser.add_argument("--img_size", type=int, default=28, help="no of epochs")
parser.add_argument("--channels", type=int, default=1, help="no of epochs")
parser.add_argument("--sample_interval", type=int, default=400, help="no of epochs")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(opt.lrelu_a, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128),
            nn.Linear(128, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 128),
            nn.LeakyReLU(opt.lrelu_a, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
            # F.leaky_relu()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# LOSS function
adversarial_loss = nn.BCELoss()

# initialize networks
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda(0)
    discriminator.cuda(0)
    adversarial_loss.cuda(0)

# set optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# LOAD DATASET
os.makedirs("./data/mnist", exist_ok=True)
dataset_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist",
        download=True,
        train=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#####################
# TRAINING
#####################
for epoch in range(opt.epochs):
    for i, (imgs, _) in enumerate(dataset_loader):
        # Adversarial Ground Truths
        valid = Tensor(imgs.size(0), 1).fill_(1.0)
        fake = Tensor(imgs.size(0), 1).fill_(0.0)

        valid.requires_grad = False
        fake.requires_grad = False

        # Configure input
        real_imgs = imgs.type(Tensor)

        #################
        # Train Generator
        #################

        optimizer_G.zero_grad()

        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))

        # Generate images
        gen_imgs = generator(z)

        # Loss measure gen ability to fool dis
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        #####################
        # Train Discriminator
        #####################

        optimizer_D.zero_grad()

        # Measure D's ability
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            f"[Epoch {epoch}/{opt.epochs}] "
            f"[Batch {i} {len(dataset_loader)}] "
            f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]]"
        )
        m_gen_imgs = gen_imgs.data[:64]
        ygen = m_gen_imgs.cpu().data.numpy()
        ygen = np.squeeze(ygen, axis=1)
        # print(ygen.shape)
        img_tile(ygen, "images", epoch, i, "res", False)
        # batches_done = epoch * len(dataset_loader) + i
        # if batches_done % opt.sample_interval == 0:
        # save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
