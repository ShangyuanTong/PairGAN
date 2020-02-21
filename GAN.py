# Modified from https://raw.githubusercontent.com/pytorch/examples/master/dcgan/main.py

from __future__ import print_function
import argparse
import os
import random
import torch
import shutil
import sys
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cat')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--save_freq_epoch', default=1, type=int, help='how frequently do you want to save for epoch')
parser.add_argument('--save_freq_step', default=10000, type=int, help='how frequently do you want to save for step')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

shutil.copy(__file__, os.path.join(opt.outf, 'code.py'))

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

with open(os.path.join(opt.outf, 'command_seed.txt'), "w") as command_file:
    command_file.write(' '.join(sys.argv))
    command_file.write('\n')
    command_file.write(str(opt.manualSeed))
    command_file.write('\n')

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['cat']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
imgsize = int(opt.imageSize)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        assert imgsize % 16 == 0, "image size has to be a multiple of 16"

        main = torch.nn.Sequential()

        mult = imgsize // 8
        main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(nz, ngf * mult, 4, 1, 0, bias=False))
        main.add_module('Start-BatchNorm2d', torch.nn.BatchNorm2d(ngf * mult))
        main.add_module('Start-ReLU', torch.nn.ReLU(True))
        i = 0
        while mult > 1:
            i += 1
            mult = mult // 2
            main.add_module('Middle-ConvTranspose2d [%d]' % i, torch.nn.ConvTranspose2d(ngf * mult * 2, ngf * mult, 4, 2, 1, bias=False))
            main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(ngf * mult))
            main.add_module('Middle-ReLU [%d]' % i, torch.nn.ReLU(True))
        main.add_module('End-ConvTranspose2d', torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False))
        main.add_module('End-Tanh', torch.nn.Tanh())

        self.main = main
        self.ngpu = ngpu

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        assert imgsize % 16 == 0, "image size has to be a multiple of 16"

        main = torch.nn.Sequential()

        main.add_module('Start-Conv2d', torch.nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('Start-LeakyReLU', nn.LeakyReLU(0.2, inplace=True))

        image_size_new = imgsize // 2
        mult = 1
        i = 1
        while image_size_new > 4:
            main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(ndf * mult, ndf * mult * 2, 4, 2, 1, bias=False))
            main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(ngf * mult * 2))
            main.add_module('Middle-LeakyReLU [%d]' % i, nn.LeakyReLU(0.2, inplace=True))
            image_size_new = image_size_new // 2
            mult *= 2
            i += 1

        main.add_module('End-Conv2d', torch.nn.Conv2d(ndf * mult, 1, 4, 1, 0, bias=False))

        self.main = main
        self.ngpu = ngpu

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


criterion_BCE = nn.BCEWithLogitsLoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

generator_steps = 0

print("Starting Training Loop...")
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network
        ############################
        # train with real
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_cpu)
        errD_real = criterion_BCE(output, label)
        errD_real.backward()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion_BCE(output, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network
        ############################
        # train with fake
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion_BCE(output, label)
        errG.backward()
        optimizerG.step()

        if generator_steps == 0:
            vutils.save_image(real_cpu*.5+.5,
                '%s/real_samples.png' % opt.outf,
                normalize=False)

        generator_steps += 1

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                  % (epoch, opt.niter, i, len(dataloader), errD.item(), errG.item()))

        if generator_steps % opt.save_freq_step == 0:
            torch.save(netG.state_dict(), '%s/netG_step_%06d.pth' % (opt.outf, generator_steps))
            torch.save(netD.state_dict(), '%s/netD_step_%06d.pth' % (opt.outf, generator_steps))
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach()*.5+.5,
                    '%s/fake_samples_step_%06d.png' % (opt.outf, generator_steps),
                    normalize=False)

    # do checkpointing
    if epoch % opt.save_freq_epoch == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%06d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%06d.pth' % (opt.outf, epoch))
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach()*.5+.5,
            '%s/fake_samples_epoch_%06d.png' % (opt.outf, epoch),
            normalize=False)
