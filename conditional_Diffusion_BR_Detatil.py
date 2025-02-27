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
device = torch.device("cuda", 4)
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


class_list = ['유형5', '유형6', '유형7']
params = {'image_size': 1024,
          'lr': 5e-5,
          'beta1': 0.5,
          'beta2': 0.999,
          'batch_size': 1,
          'epochs': 1000,
          'n_classes': None,
          'data_path': '../../data/origin_type/BRDC/',
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


class CustomDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, parmas, images, label):

        self.images = images
        self.args = parmas
        self.label = label

    def trans(self, image):
        if random.random() > 0.5:
            transform = transforms.RandomHorizontalFlip(1)
            image = transform(image)

        if random.random() > 0.5:
            transform = transforms.RandomVerticalFlip(1)
            image = transform(image)

        return image

    def __getitem__(self, index):
        image = self.images[index]
        label = self.label[index]
        image = self.trans(image)
        return image, label

    def __len__(self):
        return len(self.images)


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


image_label = []
image_path = []
for i in tqdm(range(len(class_list))):
    image_list = glob(params['data_path']+class_list[i]+'/*.jpeg')
    for j in range(len(image_list)):
        image_path.append(image_list[j])
        image_label.append(i)

train_images = torch.zeros(
    (len(image_path), params['inch'], params['image_size'], params['image_size']))
for i in tqdm(range(len(image_path))):
    train_images[i] = tf(Image.open(image_path[i]).convert(
        'L').resize((params['image_size'], params['image_size'])))*2-1
train_dataset = CustomDataset(params, train_images, image_label)
dataloader = DataLoader(
    train_dataset, batch_size=params['batch_size'], shuffle=True)

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


cosineScheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
warmUpScheduler = GradualWarmupScheduler(
    optimizer=optimizer,
    multiplier=params['multiplier'],
    warm_epoch=30,
    after_scheduler=cosineScheduler,
    last_epoch=0
)
checkpoint = torch.load(
    f'../../model/conditionDiff/scratch_details/BRDC/ckpt_151_checkpoint.pt', map_location=device)
diffusion.model.load_state_dict(checkpoint['net'])
cemblayer.load_state_dict(checkpoint['cemblayer'])
optimizer.load_state_dict(checkpoint['optimizer'])
warmUpScheduler.load_state_dict(checkpoint['scheduler'])


generator = GeneratorUNet()
generator.to(device1)
generator.load_state_dict(torch.load(
    '../../model/colorization/pix2pix_r/Pix2Pix_Generator_for_Colorization_360.pt', map_location=device1))


checkpoint = 0

scaler = torch.cuda.amp.GradScaler()
topilimage = torchvision.transforms.ToPILImage()
for epc in range(151, params['epochs']):
    diffusion.model.train()
    cemblayer.train()
    total_loss = 0
    steps = 0
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for img, lab in tqdmDataLoader:
            b = img.shape[0]

            x_0 = img.to(device)
            lab = lab.to(device)
            cemb = cemblayer(lab)
            loss = diffusion.trainloss(x_0, cemb=cemb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            total_loss += loss.item()
            tqdmDataLoader.set_postfix(
                ordered_dict={
                    "epoch": epc + 1,
                    "loss: ": total_loss/steps,
                    "batch per device: ": x_0.shape[0],
                    "img shape: ": x_0.shape[1:],
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                }
            )
    warmUpScheduler.step()

    diffusion.model.eval()
    cemblayer.eval()
    # generating samples
    # The model generate 80 pictures(8 per row) each time
    # pictures of same row belong to the same class
    all_samples = []
    each_device_batch = len(class_list)
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
                genshape, 100, 0.1, 'quadratic', cemb=cemb)
        else:
            generated = diffusion.sample(genshape, cemb=cemb)
        generated = torch.cat([generated, generated, generated], dim=1)
        generated = transback(generator(generated.to(device1)))
        for i in range(len(lab)):
            img_pil = topilimage(generated[i].cpu())
            createDirectory(
                f'../../result/scratch_Detail/BRDC/{class_list[lab[i]]}')
            img_pil.save(
                f'../../result/scratch_Detail/BRDC/{class_list[lab[i]]}/{epc}.png')

        # save checkpoints
        checkpoint = {
            'net': diffusion.model.state_dict(),
            'cemblayer': cemblayer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': warmUpScheduler.state_dict()
        }
    if epc % 5 == 0:
        createDirectory(f'../../model/conditionDiff/scratch_details/BRDC/')
        torch.save(
            checkpoint, f'../../model/conditionDiff/scratch_details/BRDC/ckpt_{epc+1}_checkpoint.pt')
    torch.cuda.empty_cache()
