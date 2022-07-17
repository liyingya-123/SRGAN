import argparse
import os
import numpy as np
import math
import itertools
import sys
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=10, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10000, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

# cuda = torch.cuda.is_available() #GPU的设备声名。
# DTU的设备声明
import torch_dtu.core.dtu_model as dm
dtu_device = dm.dtu_device(0) # 定义device,dtu_devise()括号中可以自行选择使用哪张卡，同GPU一致，卡编号从0开始，默认为0。
print(dtu_device) #输出"xla:0"对应GPU的"cuda:0"

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

# 将模型提交到GPU设备上，由于DTU上无法对模型进行参数初始化和保存的模型加载
# 因此，在DTU上训练模型时需要先将在CPU上进行参数初始化，之后在进行提交到DTU上
# if cuda:
#     generator = generator.cuda()
#     discriminator = discriminator.cuda()
#     feature_extractor = feature_extractor.cuda()
#     criterion_GAN = criterion_GAN.cuda()
#     criterion_content = criterion_content.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
    discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

generator = copy.deepcopy(generator).to(dtu_device)
discriminator = copy.deepcopy(discriminator).to(dtu_device)
feature_extractor = copy.deepcopy(feature_extractor).to(dtu_device)
# criterion_GAN = copy.deepcopy(criterion_GAN).to(dtu_device)
# criterion_content = copy.deepcopy(criterion_content).to(dtu_device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor #DTU中不存在类似的数据类型

dataloader = DataLoader(
    ImageDataset("../data/%s" % opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------
prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        # imgs_lr = Variable(imgs["lr"].type(Tensor))
        # imgs_hr = Variable(imgs["hr"].type(Tensor))
        imgs_lr_cpu = torch.FloatTensor(imgs["lr"]) #为了后续保存结果。
        imgs_lr = torch.FloatTensor(imgs["lr"]).to(dtu_device)
        imgs_hr = torch.FloatTensor(imgs["hr"]).to(dtu_device)

        # Adversarial ground truths
        # valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        # fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        
        valid = Variable(torch.FloatTensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False).to(dtu_device)
        fake = Variable(torch.FloatTensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False).to(dtu_device)
        
        # ------------------
        #  Train Generators
        # ------------------

        # optimizer_G.zero_grad() #注释掉的原因是。。。

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        gen_hr_cpu = gen_hr.cpu()  #为了后续保存结果
        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        optimizer_G.zero_grad() #为什么放在后面
        loss_G.backward()
        # optimizer_G.step() #GPU的跟新，需要做如下修改
        dm.optimizer_step(optimizer_G)  #原理是。。。
        print('done Generator update!!!')
        
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # optimizer_D.zero_grad()   #注释掉的原因

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()  #换位置的作用是
        loss_D.backward()
        # optimizer_D.step() #GPU的更新
        dm.optimizer_step(optimizer_D)  # DTU的更新方式

        # --------------
        #  Log Progress
        # --------------
        # Determine approximate time left(确定大约剩余时间)
        time_left = datetime.timedelta(seconds= (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s,\n"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item(),time_left)
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            # Save image grid with upsampled inputs and SRGAN outputs
            imgs_lr = nn.functional.interpolate(imgs_lr_cpu, scale_factor=4)  #make_grid需要在CPU上运行
            gen_hr = make_grid(gen_hr_cpu, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            img_grid = torch.cat((imgs_lr, gen_hr), -1)
            save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)
