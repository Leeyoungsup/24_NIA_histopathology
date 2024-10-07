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
from conditionDiffusion_mask.unet import UnetWithMask
from conditionDiffusion_mask.embedding import ConditionalEmbedding
from conditionDiffusion_mask.utils import get_named_beta_schedule
from conditionDiffusion_mask.diffusion import GaussianDiffusion
from conditionDiffusion_mask.Scheduler import GradualWarmupScheduler
from PIL import Image
import torchvision
import torch.nn as nn
print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda", 1)
print(f"Device:\t\t{device}")
topilimage = torchvision.transforms.ToPILImage()


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


class_list = ['유형8', '유형9']
mask_list = ['NT_epithelial', 'NT_immune', 'TP_in_situ', 'TP_invasive']
params = {'image_size': 1024,
          'lr': 2e-4,
          'beta1': 0.5,
          'beta2': 0.999,
          'batch_size': 1,
          'epochs': 1000,
          'n_classes': None,
          'data_path': '../../data/normalization_type/BRIL/',
          'image_count': 5000,
          'inch': 3,
          'modch': 128,
          'outch': 3,
          'chmul': [1, 1, 2, 4, 4],
          'numres': 2,
          'dtype': torch.float32,
          'cdim': 10,
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

    def __init__(self, parmas, images, mask, label):

        self.images = images
        self.masks = mask
        self.args = parmas
        self.label = label

    def trans(self, image, mask):
        if random.random() > 0.5:
            transform = transforms.RandomHorizontalFlip(1)
            image = transform(image)
            mask = transform(mask)

        if random.random() > 0.5:
            transform = transforms.RandomVerticalFlip(1)
            image = transform(image)
            mask = transform(mask)
        return image, mask

    def __getitem__(self, index):
        image = self.images[index]
        label = self.label[index]
        mask = self.masks[index]
        image, mask = self.trans(image, mask)
        return image, mask, label

    def __len__(self):
        return len(self.images)


image_label = []
image_path = []
for i in tqdm(range(len(class_list))):
    image_list = glob(params['data_path']+class_list[i]+'/*.jpeg')
    if len(image_list) > params['image_count']:
        image_list = image_list[:params['image_count']]
    for j in range(len(image_list)):
        image_path.append(image_list[j])
        image_label.append(i)

train_images = torch.zeros(
    (len(image_path), params['inch'], params['image_size'], params['image_size']))
train_masks = torch.zeros((len(image_path), len(
    mask_list)+1, params['image_size'], params['image_size']))

for i in tqdm(range(len(image_path))):
    train_images[i] = trans(Image.open(image_path[i]).convert(
        'RGB').resize((params['image_size'], params['image_size'])))
    npy_mask = np.load(image_path[i].replace('.jpeg', '.npy'))
    for j in range(len(mask_list)+1):
        train_masks[i, j] = torch.tensor(np.where(npy_mask == j, 1, -1))


train_dataset = CustomDataset(params, train_images, train_masks, image_label)
dataloader = DataLoader(
    train_dataset, batch_size=params['batch_size'], shuffle=True)

net = UnetWithMask(in_ch=params['inch'],
                   mask_ch=len(mask_list)+1,
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


Generator = Generator(3, 3).to(device)
Generator.load_state_dict(torch.load(
    '../../model/cyclegan/G_B_29.pth', map_location=device))

checkpoint = torch.load(
    f'../../model/mask_Diffusion/BRIL/ckpt_20_checkpoint.pt', map_location=device)
diffusion.model.load_state_dict(checkpoint['net'])
cemblayer.load_state_dict(checkpoint['cemblayer'])
optimizer.load_state_dict(checkpoint['optimizer'])
warmUpScheduler.load_state_dict(checkpoint['scheduler'])
mask_tensor_list = torch.zeros((len(class_list), len(
    mask_list)+1, params['image_size'], params['image_size'])).to(device)
scaler = torch.cuda.amp.GradScaler()
for epc in range(20, params['epochs']):
    diffusion.model.train()
    cemblayer.train()
    total_loss = 0
    steps = 0
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for img, mask, lab in tqdmDataLoader:
            b = img.shape[0]

            x_0 = img.to(device)
            mask_0 = mask.to(device)
            lab = lab.to(device)

            # 조건 임베딩 계산
            cemb = cemblayer(lab)
            mask_tensor_list[lab.item()] = mask_0

            loss = diffusion.trainloss(x_0, mask=mask_0, cemb=cemb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            steps += 1
            total_loss += loss.item()
            tqdmDataLoader.set_postfix(
                ordered_dict={
                    "epoch": epc + 1,
                    "loss": total_loss / steps,
                    "batch per device": x_0.shape[0],
                    "img shape": x_0.shape[1:],
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                }
            )

    warmUpScheduler.step()
    diffusion.model.eval()
    cemblayer.eval()
    all_samples = []
    each_device_batch = len(class_list)

    with torch.no_grad():
        # Create label tensor for the class embeddings
        lab = torch.ones(len(class_list), each_device_batch // len(class_list)).type(torch.long) \
            * torch.arange(start=0, end=len(class_list)).reshape(-1, 1)
        lab = lab.reshape(-1, 1).squeeze()
        lab = lab.to(device)

        # Generate conditional embeddings from the label
        cemb = cemblayer(lab)

        # Define generation shape for the image batches
        genshape = (each_device_batch, 3,
                    params['image_size'], params['image_size'])
        # Sample images using the chosen method (DDIM or standard sampling)
        if params['ddim']:
            generated = diffusion.ddim_sample(
                genshape, 100, 0.0, 'quadratic', mask=mask_tensor_list, cemb=cemb)
        else:
            generated = diffusion.sample(genshape, mask=mask, cemb=cemb)

        # Convert the generated tensors to images and save them
        generated = transback(Generator(generated.to(device)).to(device))
        for i in range(len(lab)):
            img_pil = topilimage(torch.concat([generated[i].cpu(), (mask_tensor_list[i].cpu(
            ).argmax(dim=0)/len(mask_list)*2-1).unsqueeze(0).repeat(3, 1, 1)], dim=2))
            createDirectory(
                f'../../result/mask_Diffusion/BRIL/{class_list[lab[i]]}/')
            img_pil.save(
                f'../../result/mask_Diffusion/BRIL/{class_list[lab[i]]}/{epc}.png')

        # Save model checkpoints
        checkpoint = {
            'net': diffusion.model.state_dict(),
            'cemblayer': cemblayer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': warmUpScheduler.state_dict()
        }
        createDirectory(
            f'../../model/mask_Diffusion/BRIL/')
        torch.save(
            checkpoint, f'../../model/mask_Diffusion/BRIL/ckpt_{epc+1}_checkpoint.pt')
