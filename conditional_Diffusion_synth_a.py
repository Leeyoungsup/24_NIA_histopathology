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
device = torch.device("cuda", 5)
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


class_list = ['유형1', '유형2']

params = {'image_size': 1024,
          'lr': 2e-5,
          'beta1': 0.5,
          'beta2': 0.999,
          'batch_size': 1,
          'epochs': 1000,
          'n_classes': None,
          'data_path': '../../result/synth/STNT/',
          'image_count': 5000,
          'inch': 3,
          'modch': 128,
          'outch': 3,
          'chmul': [1, 2, 4, 4, 4],
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


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_features),
                      nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(64, output_nc, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


generator = Generator(3, 3).to(device)
generator.load_state_dict(torch.load(
    '../../model/cyclegan/G_B_29.pth', map_location=device))


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
    f'../../model/conditionDiff/color_scratch_details/STNT/ckpt_136_checkpoint.pt', map_location=device)
diffusion.model.load_state_dict(checkpoint['net'])
cemblayer.load_state_dict(checkpoint['cemblayer'])
optimizer.load_state_dict(checkpoint['optimizer'])
warmUpScheduler.load_state_dict(checkpoint['scheduler'])


checkpoint = 0

scaler = torch.cuda.amp.GradScaler()
topilimage = torchvision.transforms.ToPILImage()
diffusion.model.eval()
cemblayer.eval()

count = {key: 300000 for key in class_list}
while (True):

    # generating samples
    # The model generate 80 pictures(8 per row) each time
    # pictures of same row belong to the same class
    all_samples = []
    each_device_batch = len(class_list)*5
    with torch.no_grad():
        lab = torch.ones(len(class_list), each_device_batch // len(class_list)).type(torch.long)* torch.arange(start=1, end=len(class_list)).reshape(-1, 1)
        # lab = torch.tensor([[0, 0, 1, 4], [0, 0,1, 4]], dtype=torch.long)
        lab = lab.reshape(-1, 1).squeeze()
        lab = lab.to(device)
        cemb = cemblayer(lab)
        genshape = (len(lab), params['outch'],
                    params['image_size'], params['image_size'])
        if params['ddim']:
            generated = diffusion.ddim_sample(
                genshape, 100, 0.0, 'quadratic', cemb=cemb)
        else:
            generated = diffusion.sample(genshape, cemb=cemb)
        generated = transback(generator(generated.to(device)))
        for i in range(len(lab)):
            img_pil = topilimage(generated[i].cpu())
            createDirectory(
                params['data_path']+f'{class_list[lab[i]]}')
            img_pil.save(
                params['data_path']+f'{class_list[lab[i]]}/NIA_S_STNT_{class_list[lab[i]][2:]}_{str(count[class_list[lab[i]]]).zfill(6)}.jpeg')
            count[class_list[lab[i]]] += 1

    torch.cuda.empty_cache()
