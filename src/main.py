#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.parallel

from torch import optim
from torch.autograd import Variable
from torchvision import utils as vutils

from os.path import join
from processing import data_processor
from processing import utils
from models import gans

DATA_DIR = './data/'
RESULTS_DIR = './results'
MODELS_DIR = join(RESULTS_DIR, 'models')

def initializer():
    utils.mkdir(DATA_DIR)
    utils.mkdir(RESULTS_DIR)
    utils.mkdir(MODELS_DIR)
    return

def train(dataloader, num_epochs=25):
    netG = gans.Generator()
    netG.apply(gans.init_weights)

    netD = gans.Discriminator()
    netD.apply(gans.init_weights)

    criterion = nn.BCELoss()
    optimizerG = optim.Adam(
        netG.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999))
    optimizerD = optim.Adam(
        netD.parameters(),
        lr=0.0002,
        betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):

            netD.zero_grad()
            real, _ = data
            input = Variable(real)
            target = Variable(torch.ones(input.size()[0]))
            output = netD(input)
            errD_real = criterion(output, target)

            noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
            fake = netG(noise)
            target = Variable(torch.zeros(input.size()[0]))
            output = netD(fake.detach())
            errD_fake = criterion(output, target)

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            netG.zero_grad()
            target = Variable(torch.ones(input.size()[0]))
            output = netD(fake)
            errG = criterion(output, target)
            errG.backward()
            optimizerG.step()

            print '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'% (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0])

            if i % 100 == 0:
                vutils.save_image(
                    real,
                    '%s/real_samples.png' % './results',
                    normalize = True)

                fake = netG(noise)
                vutils.save_image(
                    fake.data,
                    '%s/fake_samples_epoch_%03d.png' % ("./results", epoch),
                    normalize = True)

    torch.save(netG.state_dict(), join(MODELS_DIR, 'generator.pth'))
    torch.save(netD.state_dict(), join(MODELS_DIR, 'discriminator.pth'))

if __name__ == '__main__':
    initializer()
    dataloader = data_processor.load_data(DATA_DIR)
    train(dataloader)
