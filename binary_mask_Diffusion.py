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
device = torch.device("cuda", 6)
device1 = torch.device("cuda", 1)
print(f"Device:\t\t{device}")
tf = transforms.ToTensor()
topilimage = torchvision.transforms.ToPILImage()


class_list = ['NT_epithelial', 'NT_immune',
              'NT_stroma', 'TP_in_situ', 'TP_invasive']
params = {'image_size': 1024,
          'lr': 2e-5,
          'beta1': 0.5,
          'beta2': 0.999,
          'batch_size': 1,
          'epochs': 1000,
          'n_classes': None,
          'data_path': '../../data/normalization_type/BRIL/**/',
          'image_count': 5000,
          'inch': 1,
          'modch': 128,
          'outch': 1,
          'chmul': [1, 2, 2, 4, 4, 8],
          'numres': 2,
          'dtype': torch.float32,
          'cdim': 1024*1024,
          'useconv': False,
          'droprate': 0.1,
          'T': 1000,
          'w': 1.8,
          'v': 0.3,
          'multiplier': 1,
          'threshold': 0.1,
          'ddim': True,
          }
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def transback(data: Tensor) -> Tensor:
    return data / 2 + 0.5


class CustomDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, parmas, images, label):

        self.images = images
        self.args = parmas
        self.label = label

    def trans(self, image, label):
        if random.random() > 0.5:
            transform = transforms.RandomHorizontalFlip(1)
            image = transform(image)
            label = transform(label)
        if random.random() > 0.5:
            transform = transforms.RandomVerticalFlip(1)
            image = transform(image)
            label = transform(label)
        return image, label

    def __getitem__(self, index):
        image = self.images[index]
        label = self.label[index]
        image, label = self.trans(image, label)
        return image, label

    def __len__(self):
        return len(self.images)


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
        # 너비와 높이가 2배씩 증가
        layers = [nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)  # 채널 레벨에서 합치기(concatenation)

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


image_list = glob(params['data_path']+'/*.jpeg')


train_images = torch.zeros(
    (len(image_list), params['inch'], params['image_size'], params['image_size']))
train_label = torch.zeros(
    (len(image_list), params['image_size'], params['image_size']))
for i in tqdm(range(len(image_list))):
    train_images[i] = tf(Image.open(image_list[i]).convert(
        'L').resize((params['image_size'], params['image_size'])))*2-1
    npy_label = np.load(image_list[i].replace(
        '/BRIL', '/BR_mask/BRIL').replace('jpeg', 'npy'))

    train_label[i] = torch.tensor(npy_label).float()

train_dataset = CustomDataset(params, train_images, train_label)
dataloader = DataLoader(
    train_dataset, batch_size=params['batch_size'], shuffle=True)

generator = GeneratorUNet()
generator.to(device1)
generator.load_state_dict(torch.load(
    '../../model/colorization/pix2pix/Pix2Pix_Generator_for_Colorization_176.pt', map_location=device1))
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
        diffusion.model.parameters()
    ),
    lr=params['lr'],
    weight_decay=1e-6
)


cosineScheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
warmUpScheduler = GradualWarmupScheduler(
    optimizer=optimizer,
    multiplier=params['multiplier'],
    warm_epoch=50,
    after_scheduler=cosineScheduler,
    last_epoch=0
)
# checkpoint=torch.load(f'../../model/conditionDiff/BRIL/ckpt_35_checkpoint.pt',map_location=device)
# diffusion.model.load_state_dict(checkpoint['net'])

checkpoint = 0

scaler = torch.cuda.amp.GradScaler()

for epc in range(params['epochs']):
    diffusion.model.train()
    total_loss = 0
    steps = 0
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for img, lab in tqdmDataLoader:
            b = img.shape[0]

            x_0 = img.to(device)
            lab = lab.to(device)
            loss = diffusion.trainloss(x_0, cemb=lab)
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
    # generating samples
    # The model generate 80 pictures(8 per row) each time
    # pictures of same row belong to the same class
    all_samples = []
    each_device_batch = len(class_list)
    count = 1
    with torch.no_grad():
        for i, (img, lab1) in enumerate(dataloader):
            if i == 0:
                lab = lab1.to(device)
            elif i < count:
                lab = torch.cat((lab, lab1.to(device)), 0).to(device)
            else:
                break
        genshape = (count, 1, params['image_size'], params['image_size'])
        if params['ddim']:
            generated = diffusion.ddim_sample(
                genshape, 100, 0.5, 'quadratic', cemb=lab)
        else:
            generated = diffusion.sample(genshape, cemb=lab)
        generated = torch.cat([generated, generated, generated], dim=1)
        generated = transback(generator(generated.to(device1)))
        lab = lab.cpu()
        for i in range(len(lab)):
            img_tensor = torch.zeros(
                (3, params['image_size'], params['image_size']))
            img_tensor[0] += torch.where(lab[i] == 1, 1, 0)
            img_tensor[1] += torch.where(lab[i] == 2, 1, 0)
            img_tensor[2] += torch.where(lab[i] == 3, 1, 0)
            img_tensor[0] += torch.where(lab[i] == 4, 1, 0)
            img_tensor[1] += torch.where(lab[i] == 4, 1, 0)
            img_tensor[1] += torch.where(lab[i] == 5, 1, 0)
            img_tensor[2] += torch.where(lab[i] == 5, 1, 0)
            img_pil = topilimage(
                torch.cat((generated[i].cpu(), img_tensor.float()), dim=2))
            img_pil.save(f'../../result/binary_mask_synth/BRIL/{epc}_{i}.png')

    # save checkpoints
        checkpoint = {
            'net': diffusion.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': warmUpScheduler.state_dict()
        }
    torch.save(
        checkpoint, f'../../model/binary_mask_synth/BRIL/ckpt_{epc+1}_checkpoint.pt')
    torch.cuda.empty_cache()
