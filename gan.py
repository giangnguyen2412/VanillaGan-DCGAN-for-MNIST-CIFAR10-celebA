import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import tensorboardX
from torchvision.models.inception import inception_v3
from scipy.stats import entropy

writer = tensorboardX.SummaryWriter(log_dir='./logs')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


class Generator(nn.Module):
    def __init__(self, model_arch='vanilla_gan', model_type='cifar10'):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True, bias=True):
            layers = [nn.Linear(in_feat, out_feat, bias=bias)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model_type = model_type
        self.model_arch = model_arch
        self.image_shape = {'mnist': (1, 32, 32),
                            'cifar10': (3, 64, 64),
                            'celebA': (3, 64, 64)}

        if self.model_type == 'mnist':
            channels = 1
            img_size = 32
        elif self.model_type == 'cifar10' or self.model_type == 'celebA':
            channels = 3
            img_size = 64
        else:
            print("Error: Not defined")
            return

        self.init_size = img_size // 4

        if self.model_arch == 'dc_gan':
            self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.models_cifar10 = nn.ModuleDict({

            'vanilla_gan':  nn.Sequential(
                            *block(opt.latent_dim, 128, normalize=False),
                            *block(128, 256),
                            *block(256, 512),
                            *block(512, 1024),
                            *block(1024, 2048),
                            *block(2048, 2048),
                            *block(2048, 1024),
                            nn.Linear(1024, 3 * 64 * 64),
                            nn.Tanh()),

            'dc_gan':       nn.Sequential(
                            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
                            nn.BatchNorm2d(512),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(256),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(128),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(True),
                            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
                            nn.Tanh())

        })

        self.models_mnist = nn.ModuleDict({

            'vanilla_gan':  nn.Sequential(
                            *block(opt.latent_dim, 128, normalize=False),
                            *block(128, 256),
                            *block(256, 512),
                            *block(512, 1024),
                            nn.Linear(1024, channels * img_size * img_size),
                            nn.Tanh()),

            'dc_gan':       nn.Sequential(
                            nn.Conv2d(opt.latent_dim, 128, 3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(128, 128, 3, stride=1, padding=1),
                            nn.BatchNorm2d(128),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(128, 64, 3, stride=1, padding=1),
                            nn.BatchNorm2d(64, 0.8),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(64, 64, 3, stride=1, padding=1),
                            nn.BatchNorm2d(64, 0.8),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(64, 32, 3, stride=1, padding=1),
                            nn.BatchNorm2d(32, 0.8),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(32, 16, 3, stride=1, padding=1),
                            nn.BatchNorm2d(16, 0.8),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(16, channels, 3, stride=1, padding=1),
                            nn.Tanh())

        })

        self.models_celebA = nn.ModuleDict({

            'dc_gan':       nn.Sequential(
                            # input is Z, going into a convolution
                            nn.ConvTranspose2d(opt.latent_dim, 64 * 8, 4, 1, 0, bias=False),
                            nn.BatchNorm2d(64 * 8),
                            nn.ReLU(True),
                            # state size. (ngf*8) x 4 x 4
                            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(64 * 4),
                            nn.ReLU(True),
                            # state size. (ngf*4) x 8 x 8
                            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(64 * 2),
                            nn.ReLU(True),
                            # state size. (ngf*2) x 16 x 16
                            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(64),
                            nn.ReLU(True),
                            # state size. (ngf) x 32 x 32
                            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                            nn.Tanh()
                            # state size. (nc) x 64 x 64
            )

        })

        self.model = {'mnist': self.models_mnist,
                      'cifar10': self.models_cifar10,
                      'celebA': self.models_celebA}

    def forward(self, z):
        if self.model_arch == 'vanilla_gan':
            z = z.view(z.size(0), -1)
        img = self.model[self.model_type][self.model_arch](z)
        if self.model_arch == 'vanilla_gan':
            img = img.view(img.size(0), *self.image_shape[self.model_type])
        return img


class Discriminator(nn.Module):
    def __init__(self, model_arch='dc_gan', model_type='cifar10'):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model_type = model_type
        self.model_arch = model_arch
        self.image_shape = {'mnist': (1, 32, 32),
                            'cifar10': (3, 64, 64),
                            'celebA': (3, 64, 64)}

        if self.model_type == 'mnist':
            channels = 1
            img_size = 32
        elif self.model_type == 'cifar10' or self.model_type == 'celebA':
            channels = 3
            img_size = 64
        else:
            print("Channel size is not defined")
            return

        self.init_size = img_size // 4

        if self.model_arch == 'dc_gan':
            ds_size = img_size // 2 ** 4
            self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

        self.models_cifar10 = nn.ModuleDict({

            'vanilla_gan':  nn.Sequential(
                            nn.Linear(3 * 64 * 64, 512),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(512, 256),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(256, 256),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(256, 1),
                            nn.Sigmoid()),

            'dc_gan':       nn.Sequential(
                            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(128),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(256),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(512),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                            nn.Sigmoid())
        })

        self.models_mnist = nn.ModuleDict({

            'vanilla_gan':  nn.Sequential(
                            nn.Linear(channels * img_size * img_size, 512),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(512, 256),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(256, 1),
                            nn.Sigmoid()),

            'dc_gan':       nn.Sequential(
                            *discriminator_block(channels, 16, bn=False),
                            *discriminator_block(16, 32),
                            *discriminator_block(32, 64),
                            *discriminator_block(64, 128))
        })

        self.models_celebA = nn.ModuleDict({

            'dc_gan':       nn.Sequential(

                            # input is (nc) x 64 x 64
                            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                            nn.LeakyReLU(0.2, inplace=True),
                            # state size. (ndf) x 32 x 32
                            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(64 * 2),
                            nn.LeakyReLU(0.2, inplace=True),
                            # state size. (ndf*2) x 16 x 16
                            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(64 * 4),
                            nn.LeakyReLU(0.2, inplace=True),
                            # state size. (ndf*4) x 8 x 8
                            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                            nn.BatchNorm2d(64 * 8),
                            nn.LeakyReLU(0.2, inplace=True),
                            # state size. (ndf*8) x 4 x 4
                            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
                            nn.Sigmoid()

            )
        })

        self.model = {'mnist': self.models_mnist,
                      'cifar10': self.models_cifar10,
                      'celebA': self.models_celebA}

    def forward(self, img):
        if self.model_arch == 'dc_gan':
            out = self.model[self.model_type][self.model_arch](img)
            if self.model_type == 'mnist':
                out = out.view(out.shape[0], -1)
                return self.adv_layer(out)
            return out

        elif self.model_arch == 'vanilla_gan':
            out = img.view(img.shape[0], -1)
            validity = self.model[self.model_type][self.model_arch](out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
model_arch = 'vanilla_gan'
model_type = 'cifar10'
generator = Generator(model_arch=model_arch, model_type=model_type)
discriminator = Discriminator(model_arch=model_arch, model_type=model_type)

os.makedirs("{}_{}".format(model_arch, model_type), exist_ok=True)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader_mnist = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)


class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)


dataloader_cifar = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        "../../data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

cifar = datasets.CIFAR10(
        "../../data/cifar10",
        train=True,
        download=True,
        transform=transforms.Compose([
                    transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
)

dataroot = "data/celeba"

# We can use an image folder dataset the way we have it setup.
# Create the dataset
celebA_dataset = datasets.ImageFolder(root=dataroot, transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
dataloader_celebA = torch.utils.data.DataLoader(celebA_dataset, batch_size=128, shuffle=True, num_workers=2)

import matplotlib.pyplot as plt
import torchvision.utils as vutils

device = torch.device("cuda:0" if cuda else "cpu")
# Plot some training images
real_batch = next(iter(dataloader_celebA))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.show()


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
if model_type == 'mnist':
    dataloader = dataloader_mnist
elif model_type == 'cifar10':
    dataloader = dataloader_cifar
elif model_type == 'celebA':
    dataloader = dataloader_celebA

gn = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        for j in range(3):
            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = torch.randn(imgs.shape[0], opt.latent_dim, 1, 1, device=device)
            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        cache = discriminator(real_imgs)
        real_loss = adversarial_loss(cache, valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        gn += 1
        writer.add_scalars(main_tag='loss', tag_scalar_dict={'g_loss':g_loss, 'd_loss':d_loss}, global_step=gn)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )
        "{}_{}".format(model_arch, model_type)
        save_image(gen_imgs.data[:50], "{}_{}/epoch_{}.png".format(model_arch, model_type, epoch), nrow=10, normalize=True)

        # IgnoreLabelDataset(cifar)
        print("Calculating Inception Score ...")
        score = inception_score(gen_imgs, cuda=True, batch_size=32, resize=True, splits=10)
        print(score)
