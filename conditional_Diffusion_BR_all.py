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
print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda", 4)
print(f"Device:\t\t{device}")

class_list = ['BRNT', 'BRLC', 'BRIL', 'BRID', 'BRDC']
params = {'image_size': 1024,
          'lr': 1e-5,
          'beta1': 0.5,
          'beta2': 0.999,
          'batch_size': 1,
          'epochs': 1000,
          'n_classes': None,
          'data_path': '../../data/NIA/',
          'image_count': 10000,
          'inch': 3,
          'modch': 32,
          'outch': 3,
          'chmul': [1, 2, 4, 8, 16, 32, 64],
          'numres': 2,
          'dtype': torch.float32,
          'cdim': 10,
          'useconv': False,
          'droprate': 0.1,
          'T': 1000,
          'w': 1.8,
          'v': 0.3,
          'multiplier': 2.5,
          'threshold': 0.1,
          'ddim': True,
          }


def transback(data: Tensor) -> Tensor:
    return data / 2 + 0.5


class CustomDataset(Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, parmas, images, label, count_list):
        self.count_list = count_list
        self.images = images
        self.args = parmas
        self.label = label
        self.trans1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def trans(self, image):
        if random.random() > 0.5:
            transform = transforms.RandomHorizontalFlip(1)
            image = transform(image)

        if random.random() > 0.5:
            transform = transforms.RandomVerticalFlip(1)
            image = transform(image)

        return image

    def __getitem__(self, index):
        if index//10000 == 0:
            start = 0
            ind = random.randint(start, self.count_list[index//10000]-1)
            image = self.trans1(Image.open(self.images[ind]).convert(
                'RGB').resize((params['image_size'], params['image_size'])))
            label = self.label[ind]
            image = self.trans(image)
        elif index//10000 == 1:
            start = self.count_list[0]
            ind = random.randint(start, start+self.count_list[index//10000]-1)
            image = self.trans1(Image.open(self.images[ind]).convert(
                'RGB').resize((params['image_size'], params['image_size'])))
            label = self.label[ind]
            image = self.trans(image)
        elif index//10000 == 2:
            start = self.count_list[0]+self.count_list[1]
            ind = random.randint(start, start+self.count_list[index//10000]-1)
            image = self.trans1(Image.open(self.images[ind]).convert(
                'RGB').resize((params['image_size'], params['image_size'])))
            label = self.label[ind]
        elif index//10000 == 3:
            start = self.count_list[0]+self.count_list[1]+self.count_list[2]
            ind = random.randint(start, start+self.count_list[index//10000]-1)
            image = self.trans1(Image.open(self.images[ind]).convert(
                'RGB').resize((params['image_size'], params['image_size'])))
            label = self.label[ind]
        elif index//10000 == 4:
            start = self.count_list[0]+self.count_list[1] + \
                self.count_list[2]+self.count_list[3]
            ind = random.randint(start, start+self.count_list[index//10000]-1)
            image = self.trans1(Image.open(self.images[ind]).convert(
                'RGB').resize((params['image_size'], params['image_size'])))
            label = self.label[ind]

        return image, label

    def __len__(self):
        return 50000


image_label = []
image_path = []
count_list = []
for i in tqdm(range(len(class_list))):
    image_list = glob(params['data_path']+class_list[i]+'/*.jpeg')
    count_list.append(len(image_list))
    for j in range(len(image_list)):
        image_path.append(image_list[j])
        image_label.append(i)


train_dataset = CustomDataset(params, image_path, image_label, count_list)
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
    weight_decay=1e-4
)

cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer,
    T_max=params['epochs']/100,
    eta_min=0,
    last_epoch=-1
)
warmUpScheduler = GradualWarmupScheduler(
    optimizer=optimizer,
    multiplier=params['multiplier'],
    warm_epoch=params['epochs'] // 10,
    after_scheduler=cosineScheduler,
    last_epoch=0
)
checkpoint = torch.load(
    f'../../model/conditionDiff/BR/ckpt_22_checkpoint.pt', map_location=device)
diffusion.model.load_state_dict(checkpoint['net'])
cemblayer.load_state_dict(checkpoint['cemblayer'])
optimizer.load_state_dict(checkpoint['optimizer'])
warmUpScheduler.load_state_dict(checkpoint['scheduler'])
checkpoint = 0
topilimage = torchvision.transforms.ToPILImage()
scaler = torch.cuda.amp.GradScaler()
for epc in range(22, params['epochs']):
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
            cemb[np.where(np.random.rand(b) < params['threshold'])] = 0
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
    all_samples = []
    each_device_batch = len(class_list)
    with torch.no_grad():
        lab = torch.ones(len(class_list), each_device_batch // len(class_list)).type(torch.long) \
            * torch.arange(start=0, end=len(class_list)).reshape(-1, 1)
        lab = lab.reshape(-1, 1).squeeze()
        lab = lab.to(device)
        cemb = cemblayer(lab)
        genshape = (each_device_batch, 3,
                    params['image_size'], params['image_size'])
        if params['ddim']:
            generated = diffusion.ddim_sample(
                genshape, 50, 0, 'linear', cemb=cemb)
        else:
            generated = diffusion.sample(genshape, cemb=cemb)
        generated = transback(generated)
        for i in range(len(lab)):
            img_pil = topilimage(generated[i].cpu())
            img_pil.save(f'../../result/BR/{class_list[lab[i]]}/{epc}.png')

        # save checkpoints
        checkpoint = {
            'net': diffusion.model.state_dict(),
            'cemblayer': cemblayer.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': warmUpScheduler.state_dict()
        }
        torch.save(
            checkpoint, f'../../model/conditionDiff/BR/ckpt_{epc+1}_checkpoint.pt')
    torch.cuda.empty_cache()
