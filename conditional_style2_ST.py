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
import styleGAN.networks_stylegan2 as stylegan
import styleGAN.loss as style_loss
import torch.optim as optim
import torch.nn.functional as F
print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda", 4)
print(f"Device:\t\t{device}")
# class_list=['유형1','유형2','유형3','유형4','유형5','유형6','유형7','유형8','유형9','유형10','유형11','유형12','유형13','유형14']
class_list = ['STNT', 'STDI', 'STMX', 'STIN']
params = {'image_size': 1024,
          'lr': 1e-5,
          'beta1': 0.5,
          'beta2': 0.999,
          'batch_size': 1,
          'epochs': 1000,
          'n_classes': None,
          'data_path': '../../data/NIA/',
          'image_count': 5000,
          'inch': 3,
          'modch': 32,
          'outch': 3,
          'chmul': [1, 2, 4, 8, 16, 32, 32],
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


trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def transback(data: Tensor) -> Tensor:
    return data / 2 + 0.5

# class CustomDataset(Dataset):
#     """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
#     def __init__(self,parmas, images,label):

#         self.images = images
#         self.args=parmas
#         self.label=label

#     def trans(self,image):
#         if random.random() > 0.5:
#             transform = transforms.RandomHorizontalFlip(1)
#             image = transform(image)

#         if random.random() > 0.5:
#             transform = transforms.RandomVerticalFlip(1)
#             image = transform(image)

#         return image

#     def __getitem__(self, index):
#         image=self.images[index]
#         label=self.label[index]
#         image = self.trans(image)
#         return image,label

#     def __len__(self):
#         return len(self.images)


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

        return image, label

    def __len__(self):
        return 40000


image_count = []
image_path = []
for i in tqdm(range(len(class_list))):
    image_list = glob(params['data_path']+class_list[i]+'/*.jpeg')
    for j in range(len(image_list)):
        image_path.append(image_list[j])
    image_count.append(len(image_list))
image_label = torch.zeros((sum(image_count), 14))
count = 0
for i in range(len(image_count)):
    for j in range(image_count[i]):
        image_label[count][i] = 1
        count += 1

# train_images=torch.zeros((len(image_path),params['inch'],params['image_size'],params['image_size']))
# for i in tqdm(range(len(image_path))):
#     train_images[i]=trans(Image.open(image_path[i]).convert('RGB').resize((params['image_size'],params['image_size'])))
train_dataset = CustomDataset(params, image_path, image_label, image_count)
dataloader = DataLoader(
    train_dataset, batch_size=params['batch_size'], shuffle=True)


generator = stylegan.Generator(
    z_dim=1024,  # Input latent (Z) dimensionality
    c_dim=14,    # Conditioning label (C) dimensionality (0 = no labels)
    w_dim=512,  # Intermediate latent (W) dimensionality
    img_resolution=params['image_size'],  # Output resolution
    img_channels=3,       # Number of output color channels (3 for RGB)
).to(device)

discriminator = stylegan.Discriminator(
    # Conditioning label (C) dimensionality (0 = no labels)
    c_dim=14,
    img_resolution=params['image_size'],  # Input resolution
    img_channels=3,       # Number of input color channels (3 for RGB)
    architecture='resnet',  # Architecture: 'orig', 'skip', 'resnet'
    channel_base=32768,   # Overall multiplier for the number of channels
    channel_max=512,      # Maximum number of channels in any layer
    num_fp16_res=4,       # Use FP16 for the 4 highest resolutions
    # Clamp the output of convolution layers to +-X, None = disable clamping
    conv_clamp=None,
    cmap_dim=None,        # Dimensionality of mapped conditioning label, None = default
).to(device)


# Generator와 Discriminator의 학습률(Learning rate) 설정

# Beta1과 Beta2는 일반적으로 0.0과 0.99로 설정됩니다.
beta1 = 0.0
beta2 = 0.99

# Generator Optimizer
g_optimizer = optim.Adam(generator.parameters(), lr=2e-5, betas=(beta1, beta2))

# Discriminator Optimizer
d_optimizer = optim.Adam(discriminator.parameters(),
                         lr=2e-4, betas=(beta1, beta2))


def train_discriminator_loss(discriminator, generator, real_images, labels, z, device, r1_gamma, blur_init_sigma, blur_fade_kimg, augment_pipe, cur_nimg):
    # 진짜와 가짜 이미지에 대한 예측
    real_pred = discriminator(real_images, labels)
    fake_images = generator(z, labels)
    fake_pred = discriminator(fake_images.detach(), labels)  # labels 추가

    # 손실 계산
    loss_real = torch.nn.functional.softplus(-real_pred)
    loss_fake = torch.nn.functional.softplus(fake_pred)
    d_loss_val = loss_real + loss_fake

    # R1 regularization
    if r1_gamma > 0:
        real_images.requires_grad = True
        real_pred = discriminator(real_images, labels)  # labels 추가
        r1_grads = torch.autograd.grad(outputs=real_pred.sum(
        ), inputs=real_images, create_graph=True, allow_unused=False)[0]
        r1_penalty = r1_grads.square().sum([1, 2, 3]).mean()
        r1_loss = r1_penalty * (r1_gamma / 2)
        d_loss_val += r1_loss

    return d_loss_val.mean()


def train_generator_loss(generator, discriminator, z, labels, pl_weight, pl_mean, pl_decay, pl_no_weight_grad):
    # 가짜 이미지에 대한 예측
    fake_images = generator(z, labels)
    fake_pred = discriminator(fake_images, labels)  # labels 추가

    # 손실 계산
    g_loss_val = torch.nn.functional.softplus(-fake_pred)

    # Path length regularization
    if pl_weight > 0:
        pl_noise = torch.randn_like(
            fake_images) / np.sqrt(fake_images.shape[2] * fake_images.shape[3])
        pl_grads = torch.autograd.grad(outputs=(
            fake_images * pl_noise).sum(), inputs=z, create_graph=True, retain_graph=True)[0]

        # pl_grads의 크기를 기반으로 차원을 조정합니다.
        # (1, 1024) -> sum over dim 1 and then sqrt
        pl_lengths = pl_grads.square().sum(dim=1).sqrt()
        pl_mean = pl_mean.lerp(pl_lengths.mean(), pl_decay)
        pl_penalty = (pl_lengths - pl_mean).square()
        g_loss_val += pl_penalty * pl_weight

    return g_loss_val.mean()


checkpoint = torch.load(
    f'../../model/styleGan2/STNT/checkpoint_epoch_1.pt', map_location=device)
generator.load_state_dict(checkpoint['generator_state_dict'])
discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

checkpoint = 0

r1_gamma = 10  # R1 정규화 강도를 더 낮춤
pl_weight = 1.0  # Path Length 정규화 강도를 크게 낮춤
for epc in range(1, params['epochs']):
    gloss_total = 0
    dloss_total = 0
    step = 0
    generator.train()
    discriminator.train()
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for img, lab in tqdmDataLoader:
            real_images = img.to(device)
            labels = lab.to(device)
            z = torch.randn(params['batch_size'], 1024,
                            device=device, requires_grad=True)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            d_optimizer.zero_grad()

            # 손실 계산
            d_loss = train_discriminator_loss(
                discriminator=discriminator,
                generator=generator,
                real_images=real_images,
                labels=labels,
                z=z,
                device=device,
                r1_gamma=r1_gamma,  # R1 정규화 강도 적용
                blur_init_sigma=0,
                blur_fade_kimg=1000,
                augment_pipe=None,
                cur_nimg=step
            )
            d_loss.backward()
            d_optimizer.step()
            dloss_total += d_loss.item()

            # -----------------
            #  Train Generator
            # -----------------
            g_optimizer.zero_grad()
            g_loss = train_generator_loss(
                generator=generator,
                discriminator=discriminator,
                z=z,
                labels=labels,
                pl_weight=pl_weight,  # Path Length 정규화 강도 적용
                pl_mean=torch.zeros([]).to(device),
                pl_decay=0.01,
                pl_no_weight_grad=False
            )

            g_loss.backward()  # retain_graph를 사용하지 않고 역전파
            g_optimizer.step()
            gloss_total += g_loss.item()

            step += 1

            tqdmDataLoader.set_postfix(
                ordered_dict={
                    "epoch": epc + 1,
                    "gloss": gloss_total / step,
                    "dloss": dloss_total / step,
                    "batch per device": real_images.shape[0],
                    "img shape": real_images.shape[1:],
                }
            )
 # 이미지 생성 및 저장
    generator.eval()  # 평가 모드로 전환
    with torch.no_grad():
        for cls_idx, cls_name in enumerate(class_list):
            z = torch.randn(1, 1024, device=device)
            labels = torch.zeros((1, 14), device=device)
            labels[0, cls_idx] = 1
            generated_images = generator(z, labels)
            save_image(transback(generated_images),
                       f'../../result/styleGan2/STNT/{cls_name}/generated_images_epoch_{epc+1}.png', nrow=4)

    # 모델 저장
    checkpoint = {
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'g_optimizer_state_dict': g_optimizer.state_dict(),
        'd_optimizer_state_dict': d_optimizer.state_dict(),
        'epoch': epc + 1
    }
    torch.save(
        checkpoint, f'../../model/styleGan2/STNT/checkpoint_epoch_{epc+1}.pt')

    # 학습 모드로 다시 전환
    generator.train()

    torch.cuda.empty_cache()
