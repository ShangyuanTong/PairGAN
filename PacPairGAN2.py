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
parser.add_argument('--ngf', type=int, default=32)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netDu', default='', help="path to netDu (to continue training)")
parser.add_argument('--netDb', default='', help="path to netDb (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--wait_steps', default=500, type=int, help='number of steps for waiting to anneal')
parser.add_argument('--anneal_steps', default=500, type=int, help='number of steps for annealing')
parser.add_argument('--alpha_init', default=1.0, type=float, help='alpha init value')
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


class Discriminator_unary(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_unary, self).__init__()
        assert imgsize % 16 == 0, "image size has to be a multiple of 16"

        main = torch.nn.Sequential()

        main.add_module('Start-Conv2d', torch.nn.Conv2d(nc * 2, ndf, 4, 2, 1, bias=False))
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

        main.add_module('End-Conv2d', torch.nn.Conv2d(ndf * mult, 2, 4, 1, 0, bias=False))

        self.main = main
        self.ngpu = ngpu

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 2)


netDu = Discriminator_unary(ngpu).to(device)
netDu.apply(weights_init)
if opt.netDu != '':
    netDu.load_state_dict(torch.load(opt.netDu))
print(netDu)


class Discriminator_binary(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator_binary, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(2, 16),
            nn.SELU(inplace=True),
            nn.Linear(16, 1)
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netDb = Discriminator_binary(ngpu).to(device)
netDb.apply(weights_init)
if opt.netDb != '':
    netDb.load_state_dict(torch.load(opt.netDb))
print(netDb)


criterion_BCE = nn.BCEWithLogitsLoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
same_label = 1
diff_label = 0

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(list(netDu.parameters())+list(netDb.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

generator_steps = 0

print("Starting Training Loop...")
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        # annealing
        if generator_steps < opt.wait_steps:
            alpha = opt.alpha_init
        elif generator_steps < opt.wait_steps + opt.anneal_steps:
            alpha = opt.alpha_init * (1. - (generator_steps - opt.wait_steps) / opt.anneal_steps)
        else:
            alpha = 0.
        ############################
        # (1) Update D network
        ############################
        # prepare
        netDu.zero_grad()
        netDb.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = int(real_cpu.size(0) / 6)
        same_labels = torch.full((batch_size,), same_label, device=device)
        diff_labels = torch.full((batch_size,), diff_label, device=device)
        real1 = torch.cat((real_cpu[:batch_size], real_cpu[batch_size:batch_size*2]), 1)
        real2 = torch.cat((real_cpu[batch_size*2:batch_size*3], real_cpu[batch_size*3:batch_size*4]), 1)
        real3 = torch.cat((real_cpu[batch_size*4:batch_size*5], real_cpu[batch_size*5:batch_size*6]), 1)
        noise1 = torch.randn(batch_size, nz, 1, 1, device=device)
        noise2 = torch.randn(batch_size, nz, 1, 1, device=device)
        noise3 = torch.randn(batch_size, nz, 1, 1, device=device)
        noise4 = torch.randn(batch_size, nz, 1, 1, device=device)
        noise5 = torch.randn(batch_size, nz, 1, 1, device=device)
        noise6 = torch.randn(batch_size, nz, 1, 1, device=device)
        fake1 = torch.cat((netG(noise1), netG(noise2)), 1)
        fake2 = torch.cat((netG(noise3), netG(noise4)), 1)
        fake3 = torch.cat((netG(noise5), netG(noise6)), 1)

        # train with same
        same_real_out = netDb(netDu(real1) + torch.mean(netDu(real2)))
        same_fake_out = netDb(torch.mean(netDu(fake1.detach())) + netDu(fake2.detach()))
        errD_same_real = criterion_BCE(same_real_out, same_labels)
        errD_same_fake = criterion_BCE(same_fake_out, same_labels)

        # train with diff
        diff_out = netDb(netDu(real3) + torch.mean(netDu(fake3.detach())))
        errD_diff = criterion_BCE(diff_out, diff_labels)

        # update D
        errD = errD_same_real + errD_same_fake + 2 * errD_diff
        errD.backward()
        optimizerD.step()

        ############################
        # (2) Update G network
        ############################
        netG.zero_grad()
        # train with same fake
        same_fake_out = netDb(torch.mean(netDu(fake1)) + netDu(fake2))
        errG_same_fake_diff = criterion_BCE(same_fake_out, diff_labels)
        errG_same_fake_same = criterion_BCE(same_fake_out, same_labels)

        # train with diff
        diff_out = netDb(netDu(real3) + torch.mean(netDu(fake3)))
        errG_diff = criterion_BCE(diff_out, same_labels)

        # update G
        errG = 2 * errG_diff + errG_same_fake_diff * alpha - errG_same_fake_same * (1 - alpha)
        errG.backward()
        optimizerG.step()

        if generator_steps == 0:
            vutils.save_image(real_cpu*.5+.5,
                '%s/real_samples.png' % opt.outf,
                normalize=False)

        generator_steps += 1

        if i % 50 == 0:
            print('[%d/%d][%d/%d]\talpha: %.4f\tLoss_D: %.4f\tLoss_G: %.4f\t'
                  % (epoch, opt.niter, i, len(dataloader), alpha, errD.item(), errG.item()))

        if generator_steps % opt.save_freq_step == 0:
            torch.save(netG.state_dict(), '%s/netG_step_%06d.pth' % (opt.outf, generator_steps))
            torch.save(netDu.state_dict(), '%s/netDu_step_%06d.pth' % (opt.outf, generator_steps))
            torch.save(netDb.state_dict(), '%s/netDb_step_%06d.pth' % (opt.outf, generator_steps))
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach()*.5+.5,
                    '%s/fake_samples_step_%06d.png' % (opt.outf, generator_steps),
                    normalize=False)

    # do checkpointing
    if epoch % opt.save_freq_epoch == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%06d.pth' % (opt.outf, epoch))
        torch.save(netDu.state_dict(), '%s/netDu_epoch_%06d.pth' % (opt.outf, epoch))
        torch.save(netDb.state_dict(), '%s/netDb_epoch_%06d.pth' % (opt.outf, epoch))
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach()*.5+.5,
            '%s/fake_samples_epoch_%06d.png' % (opt.outf, epoch),
            normalize=False)
