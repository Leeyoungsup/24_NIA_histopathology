import pytorch_model_summary as tms
import os
import torch
import argparse
import itertools
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from glob import glob
from torch.utils.data.distributed import DistributedSampler
import random
from conditionDiffusion.unet import Unet
from conditionDiffusion.embedding import ConditionalEmbedding
from conditionDiffusion.utils import get_named_beta_schedule
from conditionDiffusion.diffusion import GaussianDiffusion
from conditionDiffusion.Scheduler import GradualWarmupScheduler
from PIL import Image
import torchvision
import torch.nn as nn
print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda", 0)
device1 = torch.device("cuda", 5)
print(f"Device:\t\t{device}")


def createDirectory(directory):
    """_summary_
        create Directory
    Args:
        directory (string): file_path
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


class_list = ['유형10', '유형11', '유형12', '유형13', '유형14', '유형15']
params = {'image_size': 1024,
          'lr': 5e-5,
          'beta1': 0.5,
          'beta2': 0.999,
          'batch_size': 1,
          'epochs': 1000,
          'n_classes': None,
          'data_path': '../../result/synth/BRID/',
          'image_count': 5000,
          'inch': 1,
          'modch': 128,
          'outch': 1,
          'chmul': [1, 1, 2, 2, 4, 4, 8],
          'numres': 2,
          'dtype': torch.float32,
          'cdim': 256,
          'useconv': False,
          'droprate': 0.1,
          'T': 1000,
          'w': 1.8,
          'v': 0.3,
          'multiplier': 1,
          'threshold': 0.02,
          'ddim': True,
          }
tf = transforms.ToTensor()


def transback(data: Tensor) -> Tensor:
    return data / 2 + 0.5


# U-Net 아키텍처의 다운 샘플링(Down Sampling) 모듈
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        # 너비와 높이가 2배씩 감소
        layers = [nn.Conv2d(in_channels, out_channels,
                            kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# U-Net 아키텍처의 업 샘플링(Up Sampling) 모듈: Skip Connection 입력 사용
class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(UNetUp, self).__init__()
        # ConvTranspose2d 대신 Upsample과 Conv2d 사용
        layers = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.Conv2d(in_channels, out_channels,
                            kernel_size=3, stride=1, padding=1),
                  nn.InstanceNorm2d(out_channels),
                  nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)  # 채널 레벨에서 합치기
        return x


# U-Net 생성자(Generator) 아키텍처
class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        # 출력: [64 x 512 x 512]
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        # 출력: [128 x 256 x 256]
        self.down2 = UNetDown(64, 128)
        # 출력: [256 x 128 x 128]
        self.down3 = UNetDown(128, 256)
        # 출력: [512 x 64 x 64]
        self.down4 = UNetDown(256, 512, dropout=0.5)
        # 출력: [512 x 32 x 32]
        self.down5 = UNetDown(512, 512, dropout=0.5)
        # 출력: [512 x 16 x 16]
        self.down6 = UNetDown(512, 512, dropout=0.5)
        # 출력: [512 x 8 x 8]
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False,
                              dropout=0.5)  # 출력: [512 x 4 x 4]

        # 출력: [1024 x 8 x 8]
        self.up1 = UNetUp(512, 512, dropout=0.5)
        # 출력: [1024 x 16 x 16]
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        # 출력: [1024 x 32 x 32]
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        # 출력: [1024 x 64 x 64]
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        # 출력: [512 x 128 x 128]
        self.up5 = UNetUp(1024, 256)
        # 출력: [256 x 256 x 256]
        self.up6 = UNetUp(512, 128)
        # 출력: [128 x 512 x 512]
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 출력: [128 x 1024 x 1024]
            nn.Conv2d(128, out_channels, kernel_size=3, stride=1,
                      padding=1),  # 출력: [3 x 1024 x 1024]
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


net = Unet(in_ch=params['inch'],
           mod_ch=params['modch'],
           out_ch=params['outch'],
           ch_mul=params['chmul'],
           num_res_blocks=params['numres'],
           cdim=params['cdim'],
           use_conv=params['useconv'],
           droprate=params['droprate'],
           dtype=params['dtype']
           ).to(device)

cemblayer = ConditionalEmbedding(
    len(class_list), params['cdim'], params['cdim']).to(device)
betas = get_named_beta_schedule(num_diffusion_timesteps=params['T'])
diffusion = GaussianDiffusion(
    dtype=params['dtype'],
    model=net,
    betas=betas,
    w=params['w'],
    v=params['v'],
    device=device
)
optimizer = torch.optim.AdamW(
    itertools.chain(
        diffusion.model.parameters(),
        cemblayer.parameters()
    ),
    lr=params['lr'],
    weight_decay=1e-6
)


cosineScheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
warmUpScheduler = GradualWarmupScheduler(
    optimizer=optimizer,
    multiplier=params['multiplier'],
    warm_epoch=100,
    after_scheduler=cosineScheduler,
    last_epoch=0
)
checkpoint = torch.load(
    f'../../model/conditionDiff/scratch_details/BRID/ckpt_221_checkpoint.pt', map_location=device)
diffusion.model.load_state_dict(checkpoint['net'])
cemblayer.load_state_dict(checkpoint['cemblayer'])
optimizer.load_state_dict(checkpoint['optimizer'])
warmUpScheduler.load_state_dict(checkpoint['scheduler'])


generator = GeneratorUNet()
generator.to(device1)
generator.load_state_dict(torch.load(
    '../../model/colorization/pix2pix_r/Pix2Pix_Generator_for_Colorization_23.pt', map_location=device1))


checkpoint = 0

scaler = torch.cuda.amp.GradScaler()
topilimage = torchvision.transforms.ToPILImage()
diffusion.model.eval()
cemblayer.eval()

count = {key: 0 for key in class_list}
while (True):

    # generating samples
    # The model generate 80 pictures(8 per row) each time
    # pictures of same row belong to the same class
    all_samples = []
    each_device_batch = len(class_list)*1
    with torch.no_grad():
        lab = torch.ones(len(class_list), each_device_batch // len(class_list)).type(torch.long) \
            * torch.arange(start=0, end=len(class_list)).reshape(-1, 1)
        lab = lab.reshape(-1, 1).squeeze()
        lab = lab.to(device)
        cemb = cemblayer(lab)
        genshape = (each_device_batch, params['outch'],
                    params['image_size'], params['image_size'])
        if params['ddim']:
            generated = diffusion.ddim_sample(
                genshape, 100, 0.0, 'quadratic', cemb=cemb)
        else:
            generated = diffusion.sample(genshape, cemb=cemb)
        generated = torch.cat([generated, generated, generated], dim=1)
        generated = transback(generator(generated.to(device1)))
        for i in range(len(lab)):
            img_pil = topilimage(generated[i].cpu())
            createDirectory(
                params['data_path']+f'{class_list[lab[i]]}')
            img_pil.save(
                params['data_path']+f'{class_list[lab[i]]}/{count[class_list[lab[i]]]}.png')
            count[class_list[lab[i]]] += 1

    torch.cuda.empty_cache()
