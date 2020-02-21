import argparse
import os
import torch
import random
import torch.nn as nn
import torch.nn.parallel
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--netG', required=True, help="path to netG")
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--outf', default='.', help='folder to output images')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--imgnum', type=int, default=50000, help='number of images to generate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--color', type=int, default=3, help='number of color channels')

opt = parser.parse_args()
device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
imgsize = int(opt.imageSize)
nc = int(opt.color)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

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
netG.load_state_dict(torch.load(opt.netG))


ext_curr = 0
for ext in range(int(opt.imgnum/100)):
	fake_test = netG(torch.randn(100, nz, 1, 1, device=device))
	for ext_i in range(100):
		vutils.save_image((fake_test[ext_i].data*.5)+.5, '%s/fake_samples_%05d.png' % (opt.outf, ext_curr), normalize=False, padding=0)
		ext_curr += 1
